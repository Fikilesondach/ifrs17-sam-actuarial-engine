# app.py
# ══════════════════════════════════════════════════════════════════════
# IFRS 17 / SAM Life Insurance Actuarial Valuation Engine
# Streamlit Dashboard
#
# Upload policy data → Configure assumptions → Run valuation →
# View IFRS 17 Insurance Contract Liability breakdown
# ══════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import json
import io
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

from ingestion.pipeline  import run_pipeline, load_config
from engine.assumptions  import (
    AssumptionSet, LapseAssumptions, ExpenseAssumptions,
    DiscountAssumptions, build_default_assumptions
)
from engine.ifrs17       import value_portfolio_ifrs17, value_policy_ifrs17
from engine.projector    import PolicyInput

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title   = "IFRS 17 Actuarial Engine",
    page_icon    = "📋",
    layout       = "wide",
    initial_sidebar_state = "expanded"
)

# ── Professional styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem; font-weight: 800;
        color: #1a237e; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem; color: #546e7a;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.2rem; font-weight: 700; color: #1a237e;
        border-bottom: 2px solid #1565c0;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
    .metric-positive { color: #2e7d32; font-weight: 700; }
    .metric-negative { color: #c62828; font-weight: 700; }
    .onerous-badge {
        background: #ffebee; color: #c62828;
        padding: 2px 8px; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600;
    }
    .profitable-badge {
        background: #e8f5e9; color: #2e7d32;
        padding: 2px 8px; border-radius: 4px;
        font-size: 0.8rem; font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────
COLOURS = {
    "bel"  : "#1565c0",   # Blue
    "ra"   : "#6a1b9a",   # Purple
    "fcf"  : "#00695c",   # Teal
    "csm"  : "#e65100",   # Orange
    "icl"  : "#b71c1c",   # Red
    "term" : "#1976d2",
    "wl"   : "#388e3c",
    "endow": "#f57c00",
}

PRODUCT_LABELS = {
    "TERM" : "Term Assurance",
    "WL"   : "Whole Life",
    "ENDOW": "Endowment",
}


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE — Persists data between Streamlit reruns
# ══════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "canonical_df"   : None,
        "quality_report" : None,
        "error_df"       : None,
        "summary_df"     : None,
        "results_list"   : None,
        "portfolio"      : None,
        "assumptions"    : None,
        "valuation_run"  : False,
        "mapping_config" : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="main-header">📋 IFRS 17 / SAM Life Insurance Actuarial Engine</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">IFRS 17 General Measurement Model · '
    'SAM Best Estimate Liability · '
    'Insurance Contract Liability Valuation · '
    'South African Life Insurance</p>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # ── Valuation date ─────────────────────────────────────────────────
    st.markdown("### 📅 Valuation Date")
    valuation_date = st.date_input(
        "Valuation Date",
        value=pd.Timestamp("2024-12-31"),
        help="The IFRS 17 measurement date for this valuation run."
    )
    valuation_date_str = valuation_date.strftime("%Y-%m-%d")

    st.markdown("---")

    # ── Discount assumptions ───────────────────────────────────────────
    st.markdown("### 📈 Discount Rate")
    flat_rate = st.slider(
        "ZAR Risk-Free Rate (%)",
        min_value=4.0, max_value=15.0, value=9.5, step=0.25,
        help="SARB-published risk-free rate. Used to discount future cash flows."
    ) / 100

    illiquidity = st.slider(
        "Illiquidity Premium (bps)",
        min_value=0, max_value=100, value=25, step=5,
        help="Premium added to risk-free rate for illiquid liabilities."
    ) / 10000

    st.markdown("---")

    # ── Expense assumptions ────────────────────────────────────────────
    st.markdown("### 💰 Expense Assumptions")
    per_policy = st.number_input(
        "Per-Policy Annual Expense (R)",
        min_value=100.0, max_value=5000.0,
        value=350.0, step=50.0,
        help="Annual administration cost per policy."
    )
    per_prem_pct = st.slider(
        "Per-Premium Loading (%)",
        min_value=0.0, max_value=10.0, value=2.5, step=0.25,
        help="Variable expense as percentage of premium received."
    ) / 100
    exp_inflation = st.slider(
        "Expense Inflation (%)",
        min_value=2.0, max_value=12.0, value=5.5, step=0.25,
        help="Annual growth rate applied to per-policy expenses."
    ) / 100

    st.markdown("---")

    # ── Lapse assumptions ──────────────────────────────────────────────
    st.markdown("### 🚪 Lapse Rates")
    lapse_y1 = st.slider(
        "Year 1 Lapse Rate (%)",
        min_value=1.0, max_value=30.0, value=12.0, step=0.5
    ) / 100
    lapse_lt = st.slider(
        "Long-Term Lapse Rate (%)",
        min_value=0.5, max_value=15.0, value=3.5, step=0.25
    ) / 100

    st.markdown("---")

    # ── IFRS 17 assumptions ────────────────────────────────────────────
    st.markdown("### 📊 IFRS 17 Parameters")
    ra_confidence = st.slider(
        "Risk Adjustment Confidence Level (%)",
        min_value=60, max_value=95, value=75, step=5,
        help="Percentile of loss distribution used for Risk Adjustment."
    ) / 100

    st.markdown("---")

    # ── Build assumption set from sidebar ──────────────────────────────
    def build_assumptions_from_sidebar():
        lapse_obj = LapseAssumptions(
            year_rates={
                1: lapse_y1,
                2: lapse_y1 * 0.75,
                3: lapse_y1 * 0.58,
                4: lapse_y1 * 0.46,
                5: lapse_y1 * 0.37,
            },
            long_term_rate=lapse_lt
        )
        expense_obj = ExpenseAssumptions(
            per_policy_annual = per_policy,
            per_premium_pct   = per_prem_pct,
            inflation_rate    = exp_inflation,
        )
        discount_obj = DiscountAssumptions(
            flat_rate           = flat_rate,
            illiquidity_premium = illiquidity,
        )
        return AssumptionSet(
            name                 = f"Custom — {valuation_date_str}",
            lapse                = lapse_obj,
            expenses             = expense_obj,
            discount             = discount_obj,
            ra_confidence_level  = ra_confidence,
        )

    st.session_state["assumptions"] = build_assumptions_from_sidebar()
    st.success("✓ Assumptions configured")


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
# FIND THIS LINE:
# REPLACE WITH:
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 Data Upload",
    "✅ Data Quality",
    "🔢 Run Valuation",
    "📊 Results Dashboard",
    "🔍 Policy Detail",
    "📑 Stakeholder Report",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Step 1 — Upload Policy Data</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Policy Data File")
        st.caption("Accepted formats: CSV, Excel (.xlsx), JSON")
        uploaded_file = st.file_uploader(
            "Upload policy data",
            type=["csv", "xlsx", "json"],
            help="Upload your policy administration system extract."
        )

    with col2:
        st.markdown("#### Column Mapping Configuration")
        st.caption("Upload a YAML mapping file for your source system")
        uploaded_yaml = st.file_uploader(
            "Upload mapping YAML",
            type=["yaml", "yml"],
            help="YAML file mapping your source columns to canonical names."
        )
        st.caption("No YAML? The engine will use the built-in synthetic data mapping.")

    # ── Use default mapping if none uploaded ───────────────────────────
    default_mapping_path = "mappings/synthetic_data.yaml"

    if uploaded_file is not None:
        st.markdown("---")
        run_ingestion = st.button(
            "🔄 Run Data Ingestion Pipeline",
            type="primary",
            use_container_width=True
        )

        if run_ingestion:
            with st.spinner("Running ingestion pipeline..."):
                try:
                    # Save uploaded file to temp location
                    suffix = "." + uploaded_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Save mapping YAML
                    if uploaded_yaml is not None:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".yaml", mode="w"
                        ) as ytmp:
                            ytmp.write(uploaded_yaml.read().decode("utf-8"))
                            yaml_path = ytmp.name
                    else:
                        yaml_path = default_mapping_path

                    # Run pipeline
                    df_valid, df_errors, report = run_pipeline(
                        file_path      = tmp_path,
                        mapping_path   = yaml_path,
                        valuation_date = valuation_date,
                        save_outputs   = False,
                    )

                    # Store in session
                    st.session_state["canonical_df"]   = df_valid
                    st.session_state["error_df"]       = df_errors
                    st.session_state["quality_report"] = report
                    st.session_state["valuation_run"]  = False

                    st.success(
                        f"✅ Ingestion complete — "
                        f"{report['summary']['passed_validation']:,} records "
                        f"ready for valuation."
                    )

                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")
                    st.exception(e)

    else:
        # ── Demo mode — use pre-generated synthetic data ───────────────
        st.markdown("---")
        st.info("""
        **No file uploaded yet.**

        To get started immediately, click **Load Demo Portfolio** to run
        the engine on 993 synthetic policies across Term, Whole Life,
        and Endowment products.
        """)

        if st.button("📂 Load Demo Portfolio", type="primary", use_container_width=True):
            with st.spinner("Loading synthetic portfolio..."):
                try:
                    df_valid, df_errors, report = run_pipeline(
                        file_path      = "data/synthetic/abc_life_export.csv",
                        mapping_path   = default_mapping_path,
                        valuation_date = valuation_date,
                        save_outputs   = False,
                    )
                    st.session_state["canonical_df"]   = df_valid
                    st.session_state["error_df"]       = df_errors
                    st.session_state["quality_report"] = report
                    st.session_state["valuation_run"]  = False

                    st.success(
                        f"✅ Demo portfolio loaded — "
                        f"{report['summary']['passed_validation']:,} policies ready."
                    )
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    # ── Show portfolio preview ─────────────────────────────────────────
    if st.session_state["canonical_df"] is not None:
        df = st.session_state["canonical_df"]
        st.markdown("---")
        st.markdown("#### Portfolio Preview")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Policies",   f"{len(df):,}")
        m2.metric("In-Force",
                  f"{(df['policy_status']=='IF').sum():,}")
        m3.metric("Products",
                  df["product_code"].nunique())
        m4.metric("Total Sum Assured",
                  f"R{pd.to_numeric(df['sum_assured'], errors='coerce').sum():,.0f}")

        display_cols = [
            "policy_id","product_code","age_at_valuation",
            "gender","smoker_status","sum_assured",
            "annualised_premium","remaining_term_years","policy_status"
        ]
        show_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[show_cols].head(20), use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Step 2 — Data Quality Report</p>',
                unsafe_allow_html=True)

    if st.session_state["quality_report"] is None:
        st.info("Upload and process a file in the **Data Upload** tab first.")
    else:
        report = st.session_state["quality_report"]
        s      = report["summary"]

        # ── Summary metrics ────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records",   f"{s['total_records']:,}")
        c2.metric("Passed ✓",
                  f"{s['passed_validation']:,}",
                  f"{s['pass_rate']:.1%}")
        c3.metric("Failed ✗",
                  f"{s['failed_validation']:,}",
                  delta=f"-{s['failed_validation']} excluded",
                  delta_color="inverse")
        c4.metric("Warnings ⚠",   f"{s['records_with_warnings']:,}")
        c5.metric("Anomalies ~",  f"{s['records_with_anomalies']:,}")

        # ── Pass rate gauge ────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = s["pass_rate"] * 100,
            title = {"text": "Data Quality Pass Rate (%)"},
            gauge = {
                "axis"  : {"range": [0, 100]},
                "bar"   : {"color": "#2e7d32"},
                "steps" : [
                    {"range": [0,  70], "color": "#ffcdd2"},
                    {"range": [70, 90], "color": "#fff9c4"},
                    {"range": [90,100], "color": "#c8e6c9"},
                ],
                "threshold": {
                    "line" : {"color": "red", "width": 4},
                    "thickness": 0.75, "value": 95
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        col_a, col_b = st.columns(2)

        # ── Error breakdown ────────────────────────────────────────────
        with col_a:
            st.markdown("#### Error Breakdown (Tier 1 — Fatal)")
            if report["error_breakdown"]:
                err_df = pd.DataFrame(
                    list(report["error_breakdown"].items()),
                    columns=["Error Type", "Count"]
                ).sort_values("Count", ascending=False)

                fig_err = px.bar(
                    err_df, x="Count", y="Error Type",
                    orientation="h", color_discrete_sequence=["#c62828"],
                    title="Records Excluded from Valuation"
                )
                fig_err.update_layout(height=300,
                                      yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(fig_err, use_container_width=True)
            else:
                st.success("✅ No fatal errors — all records passed Tier 1.")

        # ── Warning breakdown ──────────────────────────────────────────
        with col_b:
            st.markdown("#### Warning Breakdown (Tier 2 — Review)")
            if report["warning_breakdown"]:
                warn_df = pd.DataFrame(
                    list(report["warning_breakdown"].items()),
                    columns=["Warning Type", "Count"]
                ).sort_values("Count", ascending=False)

                fig_warn = px.bar(
                    warn_df, x="Count", y="Warning Type",
                    orientation="h", color_discrete_sequence=["#f57c00"],
                    title="Records Included but Flagged"
                )
                fig_warn.update_layout(height=300,
                                       yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(fig_warn, use_container_width=True)
            else:
                st.success("✅ No warnings raised.")

        # ── Error records download ─────────────────────────────────────
        if st.session_state["error_df"] is not None and \
           len(st.session_state["error_df"]) > 0:
            st.markdown("#### Failed Records")
            st.dataframe(
                st.session_state["error_df"].head(50),
                use_container_width=True
            )
            csv_errors = st.session_state["error_df"].to_csv(index=False)
            st.download_button(
                "⬇️ Download Error Records (CSV)",
                data=csv_errors,
                file_name="validation_errors.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — RUN VALUATION
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Step 3 — Run IFRS 17 Valuation</p>',
                unsafe_allow_html=True)

    if st.session_state["canonical_df"] is None:
        st.info("Complete the **Data Upload** step first.")
    else:
        df   = st.session_state["canonical_df"]
        inforce = df[df["policy_status"] == "IF"]

        # ── Assumption summary ─────────────────────────────────────────
        st.markdown("#### Current Assumption Set")
        a = st.session_state["assumptions"]

        col1, col2, col3 = st.columns(3)
        col1.info(f"""
**Discount**
- Risk-free rate: {flat_rate:.2%}
- Illiquidity premium: {illiquidity*10000:.0f} bps
- Effective rate: {(flat_rate + illiquidity):.2%}
        """)
        col2.info(f"""
**Expenses**
- Per-policy: R{per_policy:,.0f} p.a.
- Per-premium: {per_prem_pct:.2%}
- Inflation: {exp_inflation:.2%} p.a.
        """)
        col3.info(f"""
**Other**
- Year 1 lapse: {lapse_y1:.2%}
- Long-term lapse: {lapse_lt:.2%}
- RA confidence: {ra_confidence:.0%}
        """)

        st.markdown("---")
        st.markdown(f"""
        **Ready to value {len(inforce):,} in-force policies**
        across {inforce['product_code'].nunique()} product types
        as at **{valuation_date_str}**.
        """)

        run_valuation = st.button(
            f"🚀 Run IFRS 17 GMM Valuation ({len(inforce):,} policies)",
            type="primary",
            use_container_width=True
        )

        if run_valuation:
            progress = st.progress(0)
            status   = st.status("Running IFRS 17 valuation...", expanded=True)

            with status:
                st.write("📐 Initialising mortality tables and assumptions...")
                progress.progress(10)

                st.write(f"🔢 Projecting cash flows for {len(inforce):,} policies...")
                progress.progress(30)

                try:
                    summary_df, results_list, portfolio = value_portfolio_ifrs17(
                        canonical_df   = df,
                        assumptions    = st.session_state["assumptions"],
                        valuation_date = valuation_date_str,
                        verbose        = False,
                    )
                    progress.progress(80)

                    st.write("📊 Computing IFRS 17 components and CSM roll-forwards...")
                    st.session_state["summary_df"]    = summary_df
                    st.session_state["results_list"]  = results_list
                    st.session_state["portfolio"]     = portfolio
                    st.session_state["valuation_run"] = True
                    progress.progress(100)

                    status.update(
                        label="✅ Valuation complete!", state="complete"
                    )

                    # ── Quick KPIs ─────────────────────────────────────
                    st.markdown("---")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total ICL",
                              f"R{portfolio['total_icl']:,.0f}")
                    k2.metric("Total BEL",
                              f"R{portfolio['total_bel']:,.0f}")
                    k3.metric("Total CSM",
                              f"R{portfolio['total_csm']:,.0f}")
                    k4.metric("Onerous Contracts",
                              f"{portfolio['onerous_contracts']:,}",
                              f"{portfolio['onerous_contracts']/len(inforce):.1%} of portfolio")

                    st.info(
                        "✅ Results ready — view the **Results Dashboard** tab "
                        "for full analysis."
                    )

                except Exception as e:
                    st.error(f"❌ Valuation failed: {str(e)}")
                    st.exception(e)
                    progress.progress(0)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">IFRS 17 Valuation Results</p>',
                unsafe_allow_html=True)

    if not st.session_state["valuation_run"]:
        st.info("Run the valuation in the **Run Valuation** tab first.")
    else:
        ps  = st.session_state["portfolio"]
        sdf = st.session_state["summary_df"]

        # ── KPI Row ────────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total ICL",    f"R{ps['total_icl']/1e6:.2f}M")
        k2.metric("Total BEL",    f"R{ps['total_bel']/1e6:.2f}M")
        k3.metric("Risk Adj",     f"R{ps['total_ra']/1e6:.2f}M")
        k4.metric("Total FCF",    f"R{ps['total_fcf']/1e6:.2f}M")
        k5.metric("Total CSM",    f"R{ps['total_csm']/1e6:.2f}M")
        k6.metric("Onerous",
                  f"{ps['onerous_contracts']:,}",
                  f"Loss: R{ps['total_onerous_loss']/1e6:.2f}M",
                  delta_color="inverse")

        st.markdown("---")

        # ── Row 1: ICL Waterfall + Product breakdown ───────────────────
        row1a, row1b = st.columns([1.3, 1])

        with row1a:
            st.markdown("#### IFRS 17 Liability Waterfall")
            st.caption("How the Insurance Contract Liability builds up from its components")

            fig_wf = go.Figure(go.Waterfall(
                orientation = "v",
                measure     = ["relative","relative","relative","total"],
                x           = ["BEL","Risk Adjustment","CSM","ICL"],
                y           = [
                    ps["total_bel"] / 1e6,
                    ps["total_ra"]  / 1e6,
                    ps["total_csm"] / 1e6,
                    ps["total_icl"] / 1e6,
                ],
                text        = [
                    f"R{ps['total_bel']/1e6:.1f}M",
                    f"R{ps['total_ra']/1e6:.1f}M",
                    f"R{ps['total_csm']/1e6:.1f}M",
                    f"R{ps['total_icl']/1e6:.1f}M",
                ],
                textposition= "outside",
                connector   = {"line": {"color": "#546e7a"}},
                increasing  = {"marker": {"color": COLOURS["bel"]}},
                totals      = {"marker": {"color": COLOURS["icl"]}},
            ))
            fig_wf.update_layout(
                height=380, template="plotly_white",
                yaxis_title="Amount (R millions)",
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        with row1b:
            st.markdown("#### ICL by Product")
            st.caption("Insurance Contract Liability split across product lines")

            by_prod = ps["by_product"].reset_index()

            fig_prod = go.Figure()
            components = [
                ("total_bel", "BEL",            COLOURS["bel"]),
                ("total_ra",  "Risk Adjustment", COLOURS["ra"]),
                ("total_csm", "CSM",             COLOURS["csm"]),
            ]
            for col, label, colour in components:
                if col in by_prod.columns:
                    fig_prod.add_trace(go.Bar(
                        name   = label,
                        x      = by_prod["product_code"].map(PRODUCT_LABELS),
                        y      = by_prod[col] / 1e6,
                        marker_color = colour,
                        text   = (by_prod[col]/1e6).round(1).astype(str) + "M",
                        textposition = "inside",
                    ))

            fig_prod.update_layout(
                barmode  = "stack",
                height   = 380,
                template = "plotly_white",
                yaxis_title = "Amount (R millions)",
                legend      = dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_prod, use_container_width=True)

        st.markdown("---")

        # ── Row 2: BEL distribution + Onerous analysis ─────────────────
        row2a, row2b = st.columns(2)

        with row2a:
            st.markdown("#### BEL Distribution by Product")
            st.caption("Spread of individual policy BEL values")

            fig_box = go.Figure()
            for prod, colour in [
                ("TERM","#1976d2"),("WL","#388e3c"),("ENDOW","#f57c00")
            ]:
                subset = sdf[sdf["product_code"] == prod]["bel"]
                if len(subset) > 0:
                    fig_box.add_trace(go.Box(
                        y       = subset / 1000,
                        name    = PRODUCT_LABELS.get(prod, prod),
                        marker_color = colour,
                        boxmean = "sd",
                    ))

            fig_box.update_layout(
                height=350, template="plotly_white",
                yaxis_title="BEL per policy (R thousands)",
                showlegend=True,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with row2b:
            st.markdown("#### Onerous vs Profitable Contracts")
            st.caption("Proportion of contracts with positive FCF at inception")

            onerous_count    = ps["onerous_contracts"]
            profitable_count = ps["total_policies"] - onerous_count

            fig_donut = go.Figure(go.Pie(
                labels = ["Profitable (CSM > 0)", "Onerous (Loss recognised)"],
                values = [profitable_count, onerous_count],
                hole   = 0.50,
                marker = dict(colors=["#2e7d32", "#c62828"]),
                textinfo    = "label+percent",
                textposition= "outside",
                hovertemplate = "<b>%{label}</b><br>Policies: %{value:,}<extra></extra>"
            ))
            fig_donut.update_layout(
                height=350, template="plotly_white",
                showlegend=False,
                annotations=[dict(
                    text=f"{onerous_count/ps['total_policies']:.0%}<br>Onerous",
                    x=0.5, y=0.5, font_size=14,
                    showarrow=False, font_color="#c62828"
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")

        # ── Row 3: Age distribution + Sum assured vs BEL ──────────────
        row3a, row3b = st.columns(2)

        with row3a:
            st.markdown("#### Age at Valuation vs BEL")
            st.caption(
                "Each point is one policy — colour indicates product type"
            )
            fig_scatter = px.scatter(
                sdf,
                x="age_at_valuation", y="bel",
                color="product_code",
                color_discrete_map={
                    "TERM":"#1976d2","WL":"#388e3c","ENDOW":"#f57c00"
                },
                labels={
                    "age_at_valuation":"Age at Valuation",
                    "bel":"BEL (R)",
                    "product_code":"Product"
                },
                opacity=0.5,
                height=350,
            )
            fig_scatter.update_layout(template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with row3b:
            st.markdown("#### ICL Components — Ratio Analysis")
            st.caption("How each component contributes to total ICL by product")

            ratio_data = by_prod.copy()
            ratio_data["RA / ICL %"]  = (
                ratio_data["total_ra"]  / ratio_data["total_icl"] * 100
            ).round(1)
            ratio_data["CSM / ICL %"] = (
                ratio_data["total_csm"] / ratio_data["total_icl"] * 100
            ).round(1)
            ratio_data["BEL / ICL %"] = (
                ratio_data["total_bel"] / ratio_data["total_icl"] * 100
            ).round(1)
            ratio_data["Product"] = ratio_data["product_code"].map(PRODUCT_LABELS)

            fig_ratio = go.Figure()
            for col, label, colour in [
                ("BEL / ICL %","BEL",COLOURS["bel"]),
                ("RA / ICL %","Risk Adjustment",COLOURS["ra"]),
                ("CSM / ICL %","CSM",COLOURS["csm"]),
            ]:
                fig_ratio.add_trace(go.Bar(
                    name=label, x=ratio_data["Product"],
                    y=ratio_data[col], marker_color=colour,
                    text=ratio_data[col].astype(str)+"%",
                    textposition="inside",
                ))
            fig_ratio.update_layout(
                barmode="stack", height=350, template="plotly_white",
                yaxis_title="% of ICL",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

        st.markdown("---")

        # ── Full results table ─────────────────────────────────────────
        st.markdown("#### Full Policy Results Table")
        display_sdf = sdf[[
            "policy_id","product_code","age_at_valuation","gender",
            "sum_assured","remaining_term","bel","risk_adjustment",
            "fcf","csm","icl","is_onerous"
        ]].copy()
        display_sdf.columns = [
            "Policy ID","Product","Age","Gender",
            "Sum Assured","Term Rem.","BEL","RA",
            "FCF","CSM","ICL","Onerous"
        ]
        for col in ["BEL","RA","FCF","CSM","ICL","Sum Assured"]:
            display_sdf[col] = display_sdf[col].apply(
                lambda x: f"R{x:,.0f}"
            )

        st.dataframe(display_sdf, use_container_width=True, height=380)

        # ── Download ───────────────────────────────────────────────────
        csv_out = sdf.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Results (CSV)",
            data     = csv_out,
            file_name= f"ifrs17_results_{valuation_date_str}.csv",
            mime     = "text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — POLICY DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Policy-Level Detail</p>',
                unsafe_allow_html=True)

    if not st.session_state["valuation_run"]:
        st.info("Run the valuation first.")
    else:
        sdf     = st.session_state["summary_df"]
        results = st.session_state["results_list"]
        assumptions = st.session_state["assumptions"]

        # ── Policy selector ────────────────────────────────────────────
        col_sel, col_or = st.columns([3, 1])

        with col_sel:
            policy_ids = sdf["policy_id"].tolist()
            selected_id = st.selectbox(
                "Select Policy ID",
                options=policy_ids,
                help="Choose any policy to see its full IFRS 17 detail."
            )

        with col_or:
            st.markdown("&nbsp;")
            st.markdown("**or enter manually:**")
            manual_id = st.text_input("Policy ID", value="")
            if manual_id:
                selected_id = manual_id

        # ── Find and re-value selected policy ─────────────────────────
        if selected_id:
            canonical_df = st.session_state["canonical_df"]
            row = canonical_df[
                canonical_df["policy_id"] == selected_id
            ]

            if len(row) == 0:
                st.warning(f"Policy '{selected_id}' not found.")
            else:
                row = row.iloc[0]

                # Re-run valuation for this single policy (fast)
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
                result = value_policy_ifrs17(policy, assumptions, valuation_date_str)

                # ── Policy header ──────────────────────────────────────
                status_badge = (
                    '<span class="onerous-badge">⚠ ONEROUS</span>'
                    if result.is_onerous else
                    '<span class="profitable-badge">✓ PROFITABLE</span>'
                )
                st.markdown(
                    f"### Policy {selected_id} &nbsp; {status_badge}",
                    unsafe_allow_html=True
                )

                # ── Policy demographics ────────────────────────────────
                d1, d2, d3, d4, d5, d6 = st.columns(6)
                d1.metric("Product",
                          PRODUCT_LABELS.get(policy.product_code, policy.product_code))
                d2.metric("Age",          f"{policy.age_at_valuation:.1f}")
                d3.metric("Gender",       policy.gender)
                d4.metric("Smoker",       policy.smoker_status)
                d5.metric("Sum Assured",  f"R{policy.sum_assured:,.0f}")
                d6.metric("Ann. Premium", f"R{policy.annualised_premium:,.0f}")

                st.markdown("---")

                # ── IFRS 17 components ─────────────────────────────────
                st.markdown("#### IFRS 17 Liability Components")

                i1, i2, i3, i4, i5 = st.columns(5)
                i1.metric("BEL",
                          f"R{result.bel:,.2f}",
                          help="Best Estimate Liability: PV of future net cash flows")
                i2.metric("Risk Adjustment",
                          f"R{result.risk_adjustment:,.2f}",
                          f"{result.ra_result.ra_pct_of_bel:.1f}% of |BEL|")
                i3.metric("FCF",
                          f"R{result.fcf:,.2f}",
                          help="Fulfilment Cash Flows = BEL + RA")
                i4.metric("CSM",
                          f"R{result.csm:,.2f}",
                          help="Contractual Service Margin: unearned profit")
                i5.metric("ICL",
                          f"R{result.icl:,.2f}",
                          help="Insurance Contract Liability = FCF + CSM")

                st.markdown("---")

                # ── Cash flow projection chart ─────────────────────────
                proj_tab, csm_tab, pnl_tab = st.tabs([
                    "📈 Cash Flow Projection",
                    "📋 CSM Roll-Forward",
                    "💹 P&L Emergence",
                ])

                with proj_tab:
                    st.markdown("#### Annual Cash Flow Projection")
                    st.caption(
                        "Gross cash flows before discounting — "
                        "shows the raw expected income and outgo each year"
                    )
                    proj = result.projection.projection_table

                    fig_cf = go.Figure()
                    fig_cf.add_trace(go.Bar(
                        name="Premiums (inflow)",
                        x=proj["year"], y=proj["cf_premium"],
                        marker_color=COLOURS["bel"], opacity=0.85,
                    ))
                    fig_cf.add_trace(go.Bar(
                        name="Claims (outflow)",
                        x=proj["year"], y=-proj["cf_claim"],
                        marker_color=COLOURS["icl"], opacity=0.85,
                    ))
                    fig_cf.add_trace(go.Bar(
                        name="Expenses (outflow)",
                        x=proj["year"], y=-proj["cf_expense"],
                        marker_color=COLOURS["ra"], opacity=0.7,
                    ))
                    if proj["cf_maturity"].sum() > 0:
                        fig_cf.add_trace(go.Bar(
                            name="Maturity Benefit",
                            x=proj["year"], y=-proj["cf_maturity"],
                            marker_color=COLOURS["endow"], opacity=0.85,
                        ))
                    fig_cf.add_trace(go.Scatter(
                        name="Survival probability (l_t)",
                        x=proj["year"], y=proj["l_t"],
                        yaxis="y2", mode="lines+markers",
                        line=dict(color="#546e7a", width=2, dash="dot"),
                    ))
                    fig_cf.update_layout(
                        barmode="relative", height=420,
                        template="plotly_white",
                        xaxis_title="Policy Year",
                        yaxis_title="Cash Flow (R)",
                        yaxis2=dict(
                            title="Survival Probability",
                            overlaying="y", side="right",
                            range=[0, 1.1], showgrid=False,
                        ),
                        legend=dict(orientation="h", y=1.08),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_cf, use_container_width=True)

                    st.dataframe(
                        proj.round(4),
                        use_container_width=True, height=260
                    )

                with csm_tab:
                    st.markdown("#### CSM Roll-Forward Schedule")
                    if result.is_onerous:
                        st.warning(
                            "This contract is **onerous** — CSM is zero. "
                            "The loss of "
                            f"R{result.onerous_loss:,.2f} was recognised "
                            "immediately in P&L at inception."
                        )
                    else:
                        rf = result.csm_result.roll_forward

                        fig_csm = go.Figure()
                        fig_csm.add_trace(go.Bar(
                            name="CSM Released to P&L",
                            x=rf["year"], y=rf["csm_release"],
                            marker_color=COLOURS["csm"],
                        ))
                        fig_csm.add_trace(go.Scatter(
                            name="Closing CSM",
                            x=rf["year"], y=rf["csm_closing"],
                            mode="lines+markers",
                            line=dict(color=COLOURS["bel"], width=2.5),
                        ))
                        fig_csm.update_layout(
                            height=380, template="plotly_white",
                            xaxis_title="Policy Year",
                            yaxis_title="Amount (R)",
                            legend=dict(orientation="h", y=1.08),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_csm, use_container_width=True)
                        st.dataframe(rf.round(2),
                                     use_container_width=True, height=260)

                with pnl_tab:
                    st.markdown("#### P&L Emergence Schedule")
                    st.caption(
                        "How insurance profit is recognised under IFRS 17 — "
                        "the CSM release is the primary driver of insurance revenue"
                    )
                    pnl = result.pnl_schedule

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Bar(
                        name="Insurance Revenue",
                        x=pnl["year"], y=pnl["insurance_revenue"],
                        marker_color=COLOURS["bel"], opacity=0.85,
                    ))
                    fig_pnl.add_trace(go.Bar(
                        name="Service Expense",
                        x=pnl["year"],
                        y=-pnl["insurance_service_expense"],
                        marker_color=COLOURS["icl"], opacity=0.85,
                    ))
                    fig_pnl.add_trace(go.Scatter(
                        name="Insurance Service Result (ISR)",
                        x=pnl["year"],
                        y=pnl["insurance_service_result"],
                        mode="lines+markers",
                        line=dict(color=COLOURS["csm"], width=2.5),
                    ))
                    fig_pnl.update_layout(
                        barmode="relative", height=380,
                        template="plotly_white",
                        xaxis_title="Policy Year",
                        yaxis_title="Amount (R)",
                        legend=dict(orientation="h", y=1.08),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    st.dataframe(pnl.round(2),
                                 use_container_width=True, height=260)


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — STAKEHOLDER REPORT
# ══════════════════════════════════════════════════════════════════════
with tab6:

    if not st.session_state["valuation_run"]:
        st.info(
            "Complete the Data Upload and Run Valuation steps first. "
            "The stakeholder report generates automatically once "
            "a valuation has been run."
        )
    else:
        # ── Pull all state ─────────────────────────────────────────────
        ps          = st.session_state["portfolio"]
        sdf         = st.session_state["summary_df"]
        report      = st.session_state["quality_report"]
        assumptions = st.session_state["assumptions"]
        canonical   = st.session_state["canonical_df"]
        inforce     = canonical[canonical["policy_status"] == "IF"]
        s           = report["summary"]

        # ══════════════════════════════════════════════════════════════
        # REPORT HEADER
        # ══════════════════════════════════════════════════════════════
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        ">
            <h1 style="color:white; margin:0; font-size:1.8rem; font-weight:800;">
                IFRS 17 Actuarial Valuation Report
            </h1>
            <p style="color:#bbdefb; margin:0.4rem 0 0 0; font-size:1.0rem;">
                Life Insurance Portfolio — General Measurement Model (GMM)
            </p>
            <p style="color:#90caf9; margin:0.5rem 0 0 0; font-size:0.9rem;">
                Valuation Date: <strong style="color:white;">{valuation_date_str}</strong>
                &nbsp;|&nbsp;
                Prepared: <strong style="color:white;">{pd.Timestamp.now().strftime("%d %B %Y")}</strong>
                &nbsp;|&nbsp;
                Basis: <strong style="color:white;">Best Estimate (IFRS 17 GMM)</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Executive summary KPI strip ────────────────────────────────
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Policies Valued",
                  f"{ps['total_policies']:,}",
                  help="In-force policies included in this valuation run")
        e2.metric("Insurance Contract Liability",
                  f"R{ps['total_icl']/1e6:.1f}M",
                  help="Total ICL = FCF + CSM across all products")
        e3.metric("Contractual Service Margin",
                  f"R{ps['total_csm']/1e6:.1f}M",
                  help="Unearned profit locked in CSM at valuation date")
        e4.metric("Onerous Contracts",
                  f"{ps['onerous_contracts']:,}",
                  f"{ps['onerous_contracts']/ps['total_policies']:.1%} of portfolio",
                  delta_color="inverse")
        e5.metric("Data Quality Pass Rate",
                  f"{s['pass_rate']:.1%}",
                  help="Proportion of source records passing Tier 1 validation")

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        # SECTION 1 — DATA QUALITY
        # ══════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="
            background:#e3f2fd; border-left:5px solid #1565c0;
            padding:0.7rem 1.2rem; border-radius:0 8px 8px 0;
            margin-bottom:1rem;">
            <span style="font-size:1.15rem; font-weight:700; color:#1a237e;">
            Section 1 — Data Quality &amp; Portfolio Composition
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This section summarises the quality of the underlying policy data
        used as input to the valuation. Records are processed through a
        three-tier validation framework before reaching the projection engine.
        Any record failing Tier 1 validation is excluded from the valuation
        and disclosed separately.
        """)

        # ── Validation tier explanation ────────────────────────────────
        t1, t2, t3 = st.columns(3)
        t1.info("""
**Tier 1 — Fatal Errors**
Records excluded from valuation entirely. Typical causes: missing date of birth, invalid sum assured, duplicate policy IDs, expiry before inception.
        """)
        t2.warning("""
**Tier 2 — Warnings**
Records included but flagged for actuarial review. Typical causes: unusually high age at entry, large sum assured without reinsurance treaty, zero premium on in-force policy.
        """)
        t3.success("""
**Tier 3 — Anomalies**
Statistical outliers logged for information. Records are three or more standard deviations from portfolio mean on age, sum assured, or premium. Included in valuation.
        """)

        # ── Data quality metrics ───────────────────────────────────────
        dq1, dq2, dq3, dq4, dq5 = st.columns(5)
        dq1.metric("Source Records",     f"{s['total_records']:,}")
        dq2.metric("Passed Tier 1",      f"{s['passed_validation']:,}")
        dq3.metric("Failed Tier 1",
                   f"{s['failed_validation']:,}",
                   delta=f"-{s['failed_validation']} excluded",
                   delta_color="inverse")
        dq4.metric("Tier 2 Warnings",    f"{s['records_with_warnings']:,}")
        dq5.metric("Tier 3 Anomalies",   f"{s['records_with_anomalies']:,}")

        # ── Data quality charts ────────────────────────────────────────
        dq_col1, dq_col2 = st.columns(2)

        with dq_col1:
            # Donut: pass vs fail
            fig_dq = go.Figure(go.Pie(
                labels = ["Passed Validation","Failed Validation"],
                values = [s["passed_validation"], s["failed_validation"]],
                hole   = 0.55,
                marker = dict(colors=["#2e7d32","#c62828"]),
                textinfo     = "label+percent",
                textposition = "outside",
            ))
            fig_dq.update_layout(
                title="Validation Outcome",
                height=300, template="plotly_white", showlegend=False,
                annotations=[dict(
                    text=f"{s['pass_rate']:.1%}<br>Pass Rate",
                    x=0.5, y=0.5, font_size=14,
                    showarrow=False, font_color="#1a237e"
                )]
            )
            st.plotly_chart(fig_dq, use_container_width=True)

        with dq_col2:
            # Error breakdown bar
            if report["error_breakdown"]:
                err_df = pd.DataFrame(
                    list(report["error_breakdown"].items()),
                    columns=["Reason","Count"]
                ).sort_values("Count", ascending=True)

                fig_err = px.bar(
                    err_df, x="Count", y="Reason",
                    orientation="h",
                    color_discrete_sequence=["#c62828"],
                    title="Exclusion Reasons (Tier 1 Failures)",
                )
                fig_err.update_layout(
                    height=300, template="plotly_white",
                    yaxis={"categoryorder":"total ascending"},
                    xaxis_title="Number of Records",
                )
                st.plotly_chart(fig_err, use_container_width=True)
            else:
                st.success("No Tier 1 exclusions — all records passed validation.")

        # ── Portfolio composition ──────────────────────────────────────
        st.markdown("##### Portfolio Composition")

        comp_col1, comp_col2, comp_col3 = st.columns(3)

        with comp_col1:
            # By product
            prod_counts = canonical["product_code"].value_counts().reset_index()
            prod_counts.columns = ["Product","Count"]
            prod_counts["Product"] = prod_counts["Product"].map(
                PRODUCT_LABELS
            ).fillna(prod_counts["Product"])
            fig_prod_pie = px.pie(
                prod_counts, values="Count", names="Product",
                title="By Product",
                color_discrete_sequence=["#1976d2","#388e3c","#f57c00"],
                hole=0.4,
            )
            fig_prod_pie.update_layout(
                height=260, showlegend=True,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_prod_pie, use_container_width=True)

        with comp_col2:
            # By status
            stat_counts = canonical["policy_status"].value_counts().reset_index()
            stat_counts.columns = ["Status","Count"]
            stat_map = {"IF":"In-Force","PU":"Paid-Up","LA":"Lapsed"}
            stat_counts["Status"] = stat_counts["Status"].map(stat_map)
            fig_stat = px.pie(
                stat_counts, values="Count", names="Status",
                title="By Policy Status",
                color_discrete_sequence=["#2e7d32","#1976d2","#ef5350"],
                hole=0.4,
            )
            fig_stat.update_layout(
                height=260, showlegend=True,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_stat, use_container_width=True)

        with comp_col3:
            # Age distribution
            ages = pd.to_numeric(
                canonical["age_at_valuation"], errors="coerce"
            ).dropna()
            fig_age = px.histogram(
                ages, nbins=20,
                title="Age Distribution at Valuation",
                color_discrete_sequence=["#1565c0"],
                labels={"value":"Age","count":"Policies"},
            )
            fig_age.update_layout(
                height=260, template="plotly_white",
                showlegend=False,
                xaxis_title="Age", yaxis_title="Count",
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_age, use_container_width=True)

        # ── Portfolio statistics table ─────────────────────────────────
        st.markdown("##### Key Portfolio Statistics")
        sa_num  = pd.to_numeric(canonical["sum_assured"],       errors="coerce")
        ap_num  = pd.to_numeric(canonical["annualised_premium"], errors="coerce")
        ag_num  = pd.to_numeric(canonical["age_at_valuation"],   errors="coerce")

        stats_data = {
            "Metric": [
                "Total Policies (all statuses)",
                "In-Force Policies",
                "Total Sum Assured",
                "Mean Sum Assured",
                "Median Sum Assured",
                "Maximum Sum Assured",
                "Total Annualised Premium",
                "Mean Age at Valuation",
            ],
            "Value": [
                f"{len(canonical):,}",
                f"{(canonical['policy_status']=='IF').sum():,}",
                f"R{sa_num.sum():>20,.0f}",
                f"R{sa_num.mean():>20,.0f}",
                f"R{sa_num.median():>20,.0f}",
                f"R{sa_num.max():>20,.0f}",
                f"R{ap_num.sum():>20,.0f}",
                f"{ag_num.mean():.1f} years",
            ]
        }
        st.dataframe(
            pd.DataFrame(stats_data),
            use_container_width=True,
            hide_index=True,
            height=310,
        )

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        # SECTION 2 — BEL ANALYSIS
        # ══════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="
            background:#e8f5e9; border-left:5px solid #2e7d32;
            padding:0.7rem 1.2rem; border-radius:0 8px 8px 0;
            margin-bottom:1rem;">
            <span style="font-size:1.15rem; font-weight:700; color:#1b5e20;">
            Section 2 — Best Estimate Liability (BEL)
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        The Best Estimate Liability represents the present value of future
        expected cash outflows (claims, expenses, maturity benefits) less
        the present value of future expected cash inflows (premiums),
        discounted at the IFRS 17 discount rate of
        **{(flat_rate + illiquidity):.2%}** per annum.
        The BEL uses best estimate assumptions with no margins for prudence.

        | Component | Amount |
        |---|---|
        | PV of Future Claims | R{sdf['bel'].clip(lower=0).sum():>20,.2f} *(from projections)* |
        | Total BEL (net) | **R{ps['total_bel']:>20,.2f}** |
        | Risk Adjustment | R{ps['total_ra']:>20,.2f} |
        | **FCF (BEL + RA)** | **R{ps['total_fcf']:>20,.2f}** |
        """)

        bel_col1, bel_col2 = st.columns(2)

        with bel_col1:
            # BEL by product waterfall
            by_prod = ps["by_product"].reset_index()
            by_prod["product_label"] = by_prod["product_code"].map(PRODUCT_LABELS)

            fig_bel_prod = go.Figure()
            fig_bel_prod.add_trace(go.Bar(
                name="BEL",
                x=by_prod["product_label"],
                y=by_prod["total_bel"]/1e6,
                marker_color=COLOURS["bel"],
                text=(by_prod["total_bel"]/1e6).round(2).astype(str)+"M",
                textposition="outside",
            ))
            fig_bel_prod.add_trace(go.Bar(
                name="Risk Adjustment",
                x=by_prod["product_label"],
                y=by_prod["total_ra"]/1e6,
                marker_color=COLOURS["ra"],
                text=(by_prod["total_ra"]/1e6).round(2).astype(str)+"M",
                textposition="outside",
            ))
            fig_bel_prod.update_layout(
                barmode="group",
                title="BEL and Risk Adjustment by Product (R millions)",
                height=350, template="plotly_white",
                yaxis_title="R millions",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_bel_prod, use_container_width=True)

        with bel_col2:
            # BEL per policy histogram
            fig_bel_hist = px.histogram(
                sdf, x="bel", nbins=40, color="product_code",
                color_discrete_map={
                    "TERM":COLOURS["term"],
                    "WL":COLOURS["wl"],
                    "ENDOW":COLOURS["endow"],
                },
                labels={
                    "bel":"BEL per Policy (R)",
                    "product_code":"Product"
                },
                title="Distribution of Individual Policy BEL",
                barmode="overlay",
                opacity=0.7,
            )
            fig_bel_hist.update_layout(
                height=350, template="plotly_white",
                xaxis_title="BEL per Policy (R)",
                yaxis_title="Number of Policies",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_bel_hist, use_container_width=True)

        # ── BEL by age cohort heatmap ──────────────────────────────────
        st.markdown("##### BEL Concentration — Age vs Remaining Term")
        st.caption(
            "Average BEL per policy for each age and term combination. "
            "Darker cells indicate higher liability concentration."
        )

        sdf["age_band"] = pd.cut(
            sdf["age_at_valuation"],
            bins=[0,30,40,50,60,70,100],
            labels=["<30","30-39","40-49","50-59","60-69","70+"],
        )
        sdf["term_band"] = pd.cut(
            sdf["remaining_term"],
            bins=[0,5,10,15,20,50],
            labels=["0-5y","5-10y","10-15y","15-20y","20y+"],
        )

        pivot_bel = sdf.pivot_table(
            values="bel",
            index="age_band",
            columns="term_band",
            aggfunc="mean"
        ).fillna(0)

        fig_heat = go.Figure(go.Heatmap(
            z            = pivot_bel.values / 1000,
            x            = pivot_bel.columns.astype(str).tolist(),
            y            = pivot_bel.index.astype(str).tolist(),
            colorscale   = "Blues",
            text         = np.round(pivot_bel.values / 1000, 1),
            texttemplate = "R%{text}k",
            hovertemplate= "Age: %{y}<br>Term: %{x}<br>Avg BEL: R%{z:.1f}k<extra></extra>",
            colorbar     = dict(title="Avg BEL (R thousands)"),
        ))
        fig_heat.update_layout(
            height=300, template="plotly_white",
            xaxis_title="Remaining Term",
            yaxis_title="Age Band",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── BEL statistics by product ──────────────────────────────────
        st.markdown("##### BEL Statistics by Product")
        bel_stats = sdf.groupby("product_code")["bel"].agg([
            ("Count",    "count"),
            ("Total BEL","sum"),
            ("Mean BEL", "mean"),
            ("Median BEL","median"),
            ("Min BEL",  "min"),
            ("Max BEL",  "max"),
            ("Std Dev",  "std"),
        ]).reset_index()
        bel_stats["product_code"] = bel_stats["product_code"].map(
            PRODUCT_LABELS
        ).fillna(bel_stats["product_code"])

        for col in ["Total BEL","Mean BEL","Median BEL","Min BEL","Max BEL","Std Dev"]:
            bel_stats[col] = bel_stats[col].apply(lambda x: f"R{x:,.0f}")

        bel_stats = bel_stats.rename(columns={"product_code":"Product"})
        st.dataframe(bel_stats, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        # SECTION 3 — CSM ANALYSIS
        # ══════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="
            background:#fff3e0; border-left:5px solid #e65100;
            padding:0.7rem 1.2rem; border-radius:0 8px 8px 0;
            margin-bottom:1rem;">
            <span style="font-size:1.15rem; font-weight:700; color:#bf360c;">
            Section 3 — Contractual Service Margin (CSM)
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        The Contractual Service Margin represents the unearned profit
        embedded in the insurance contract portfolio at the valuation date.
        It is recognised in profit or loss as insurance service revenue
        in future periods as coverage is provided to policyholders.

        A contract is classified as **onerous** when its Fulfilment Cash Flows
        are positive at inception — meaning expected claims and expenses exceed
        expected premiums. Onerous contracts carry no CSM and the loss component
        is recognised immediately in profit or loss.

        | CSM Component | Amount | % of Total ICL |
        |---|---|---|
        | Total CSM — Profitable contracts | R{ps['total_csm']:>15,.2f} | {ps['total_csm']/ps['total_icl']:.1%} |
        | Onerous contract losses | R{ps['total_onerous_loss']:>15,.2f} | {ps['total_onerous_loss']/max(ps['total_icl'],1):.1%} |
        | Onerous contracts | {ps['onerous_contracts']:,} of {ps['total_policies']:,} | {ps['onerous_contracts']/ps['total_policies']:.1%} |
        """)

        csm_col1, csm_col2 = st.columns(2)

        with csm_col1:
            # CSM vs Onerous by product
            fig_csm_prod = go.Figure()
            fig_csm_prod.add_trace(go.Bar(
                name="CSM (Profitable)",
                x=by_prod["product_label"],
                y=by_prod["total_csm"]/1e6,
                marker_color=COLOURS["csm"],
                text=(by_prod["total_csm"]/1e6).round(2).astype(str)+"M",
                textposition="outside",
            ))
            if "total_onerous_loss" in by_prod.columns:
                fig_csm_prod.add_trace(go.Bar(
                    name="Onerous Loss (P&L)",
                    x=by_prod["product_label"],
                    y=by_prod["total_onerous_loss"]/1e6,
                    marker_color="#c62828",
                    text=(by_prod["total_onerous_loss"]/1e6).round(2).astype(str)+"M",
                    textposition="outside",
                ))
            fig_csm_prod.update_layout(
                barmode="group",
                title="CSM vs Onerous Loss by Product (R millions)",
                height=350, template="plotly_white",
                yaxis_title="R millions",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_csm_prod, use_container_width=True)

        with csm_col2:
            # Profitable vs onerous split per product
            fig_ono = go.Figure()
            for prod_code, prod_label in PRODUCT_LABELS.items():
                subset = sdf[sdf["product_code"] == prod_code]
                if len(subset) == 0:
                    continue
                n_ono  = subset["is_onerous"].sum()
                n_prof = len(subset) - n_ono
                fig_ono.add_trace(go.Bar(
                    name=prod_label,
                    x=["Profitable","Onerous"],
                    y=[n_prof, n_ono],
                    text=[n_prof, n_ono],
                    textposition="outside",
                ))
            fig_ono.update_layout(
                barmode="group",
                title="Profitable vs Onerous Contracts by Product",
                height=350, template="plotly_white",
                yaxis_title="Number of Policies",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_ono, use_container_width=True)

        # ── Aggregate CSM roll-forward ─────────────────────────────────
        st.markdown("##### Illustrative CSM Roll-Forward (Aggregated Portfolio)")
        st.caption(
            "This shows how the portfolio-level CSM is expected to emerge "
            "as insurance service revenue over the next 10 years, based on "
            "current coverage units and the locked-in discount rate."
        )

        results_list = st.session_state["results_list"]
        max_years    = 15

        agg_csm_rows = {yr: {"csm_release":0.0, "csm_closing":0.0, "interest":0.0}
                        for yr in range(1, max_years+1)}

        for res in results_list:
            if res.is_onerous:
                continue
            rf = res.csm_result.roll_forward
            for _, row in rf.iterrows():
                yr = int(row["year"])
                if yr <= max_years:
                    agg_csm_rows[yr]["csm_release"] += row["csm_release"]
                    agg_csm_rows[yr]["csm_closing"] += row["csm_closing"]
                    agg_csm_rows[yr]["interest"]    += row["interest_accretion"]

        agg_csm_df = pd.DataFrame(agg_csm_rows).T.reset_index()
        agg_csm_df.columns = ["Year","CSM Released","Closing CSM","Interest Accretion"]

        fig_agg_csm = make_subplots(specs=[[{"secondary_y": True}]])
        fig_agg_csm.add_trace(go.Bar(
            name="CSM Released to Revenue",
            x=agg_csm_df["Year"],
            y=agg_csm_df["CSM Released"]/1e6,
            marker_color=COLOURS["csm"],
            opacity=0.85,
            text=(agg_csm_df["CSM Released"]/1e6).round(2).astype(str)+"M",
            textposition="outside",
        ), secondary_y=False)
        fig_agg_csm.add_trace(go.Bar(
            name="Interest Accretion",
            x=agg_csm_df["Year"],
            y=agg_csm_df["Interest Accretion"]/1e6,
            marker_color=COLOURS["ra"],
            opacity=0.65,
        ), secondary_y=False)
        fig_agg_csm.add_trace(go.Scatter(
            name="Closing CSM Balance",
            x=agg_csm_df["Year"],
            y=agg_csm_df["Closing CSM"]/1e6,
            mode="lines+markers",
            line=dict(color=COLOURS["bel"], width=2.5),
        ), secondary_y=True)
        fig_agg_csm.update_layout(
            barmode="stack",
            height=400, template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08),
        )
        fig_agg_csm.update_yaxes(
            title_text="R millions (Released / Interest)", secondary_y=False
        )
        fig_agg_csm.update_yaxes(
            title_text="R millions (Closing Balance)", secondary_y=True
        )
        st.plotly_chart(fig_agg_csm, use_container_width=True)

        # ── CSM coverage by product ────────────────────────────────────
        st.markdown("##### CSM as a Percentage of ICL — By Product")
        st.caption(
            "A higher CSM/ICL ratio indicates more unearned profit "
            "relative to total liability — generally a positive indicator "
            "of portfolio profitability."
        )

        csm_ratio_data = []
        for prod_code, prod_label in PRODUCT_LABELS.items():
            subset = sdf[sdf["product_code"] == prod_code]
            if len(subset) == 0:
                continue
            total_icl = subset["icl"].sum()
            total_csm = subset["csm"].sum()
            total_bel = subset["bel"].sum()
            total_ra  = subset["risk_adjustment"].sum()
            csm_ratio_data.append({
                "Product"           : prod_label,
                "Total ICL"         : f"R{total_icl:,.0f}",
                "Total BEL"         : f"R{total_bel:,.0f}",
                "Risk Adjustment"   : f"R{total_ra:,.0f}",
                "Total CSM"         : f"R{total_csm:,.0f}",
                "CSM / ICL"         : f"{total_csm/max(total_icl,1):.1%}",
                "Onerous Count"     : int(subset["is_onerous"].sum()),
                "Profitable Count"  : int((~subset["is_onerous"]).sum()),
            })

        st.dataframe(
            pd.DataFrame(csm_ratio_data),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        # SECTION 4 — ASSUMPTION IMPACT ANALYSIS
        # ══════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="
            background:#f3e5f5; border-left:5px solid #6a1b9a;
            padding:0.7rem 1.2rem; border-radius:0 8px 8px 0;
            margin-bottom:1rem;">
            <span style="font-size:1.15rem; font-weight:700; color:#4a148c;">
            Section 4 — Assumption Sensitivity Analysis
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        Sensitivity analysis measures how the Insurance Contract Liability
        changes when individual assumptions are stressed while holding all
        others constant. This is a standard actuarial disclosure required
        by IFRS 17 paragraph 129 and is reviewed by auditors and regulators
        to assess model robustness.

        The sensitivities below apply a single stress at a time to the
        base valuation. Each stressed run is a full re-valuation of the
        in-force portfolio under the stressed assumption set.
        """)

        # ── Run sensitivities ──────────────────────────────────────────
        with st.spinner("Running sensitivity scenarios..."):

            from engine.assumptions import build_stressed_assumptions
            import copy

            base_assumptions = st.session_state["assumptions"]
            base_icl         = ps["total_icl"]
            base_bel         = ps["total_bel"]
            base_csm         = ps["total_csm"]

            scenarios = [
                # (label, assumption_modifier_fn, description)
                (
                    "Mortality +10%",
                    lambda a: _stress_mortality(a, 1.10),
                    "10% increase in qx across all ages and genders"
                ),
                (
                    "Mortality +20%",
                    lambda a: _stress_mortality(a, 1.20),
                    "20% increase in qx — SAM mortality stress benchmark"
                ),
                (
                    "Mortality -20%",
                    lambda a: _stress_mortality(a, 0.80),
                    "20% decrease in qx — longevity risk stress"
                ),
                (
                    "Lapse +50%",
                    lambda a: build_stressed_assumptions(a, "lapse_up"),
                    "50% increase in all lapse rates"
                ),
                (
                    "Lapse -50%",
                    lambda a: build_stressed_assumptions(a, "lapse_down"),
                    "50% decrease in all lapse rates"
                ),
                (
                    "Expense +10%",
                    lambda a: build_stressed_assumptions(a, "expense_up"),
                    "10% increase in all expenses + 1% additional inflation"
                ),
                (
                    "Discount Rate +100bps",
                    lambda a: _stress_discount(a, +0.01),
                    "Parallel shift up in discount curve by 100 basis points"
                ),
                (
                    "Discount Rate -100bps",
                    lambda a: _stress_discount(a, -0.01),
                    "Parallel shift down in discount curve by 100 basis points"
                ),
            ]

            # Helper stress functions
            def _stress_mortality(assumptions, multiplier):
                """Store mortality multiplier in assumption set."""
                stressed = copy.deepcopy(assumptions)
                stressed._mortality_multiplier = multiplier
                stressed.name = f"Mortality × {multiplier}"
                return stressed

            def _stress_discount(assumptions, shift):
                """Parallel shift the discount curve."""
                stressed = copy.deepcopy(assumptions)
                stressed.discount.flat_rate += shift
                if stressed.discount.term_structure:
                    stressed.discount.term_structure = {
                        k: v + shift
                        for k, v in stressed.discount.term_structure.items()
                    }
                stressed.name = f"Discount {'+' if shift>0 else ''}{shift*10000:.0f}bps"
                return stressed

            sensitivity_rows = []

            for label, stress_fn, description in scenarios:
                try:
                    stressed_a = stress_fn(base_assumptions)
                    mort_mult  = getattr(stressed_a, "_mortality_multiplier", 1.0)

                    from engine.projector import project_policy
                    from engine.ifrs17    import (
                        value_portfolio_ifrs17,
                        calculate_risk_adjustment, calculate_csm
                    )

                    stressed_summary, _, stressed_portfolio = value_portfolio_ifrs17(
                        canonical_df        = canonical,
                        assumptions         = stressed_a,
                        valuation_date      = valuation_date_str,
                        verbose             = False,
                    )

                    stressed_icl = stressed_portfolio["total_icl"]
                    stressed_bel = stressed_portfolio["total_bel"]
                    stressed_csm = stressed_portfolio["total_csm"]
                    icl_change   = stressed_icl - base_icl
                    bel_change   = stressed_bel - base_bel

                    sensitivity_rows.append({
                        "Scenario"          : label,
                        "Description"       : description,
                        "Stressed ICL"      : stressed_icl,
                        "ICL Change"        : icl_change,
                        "ICL Change %"      : icl_change / base_icl * 100,
                        "Stressed BEL"      : stressed_bel,
                        "BEL Change"        : bel_change,
                        "Stressed CSM"      : stressed_csm,
                    })

                except Exception as e:
                    sensitivity_rows.append({
                        "Scenario"     : label,
                        "Description"  : description,
                        "Stressed ICL" : None,
                        "ICL Change"   : None,
                        "ICL Change %"  : None,
                        "Stressed BEL" : None,
                        "BEL Change"   : None,
                        "Stressed CSM" : None,
                    })

        sens_df = pd.DataFrame(sensitivity_rows).dropna(subset=["ICL Change"])

        # ── Sensitivity tornado chart ──────────────────────────────────
        st.markdown("##### Sensitivity Tornado — Impact on ICL")
        st.caption(
            "Each bar shows the change in total ICL (R millions) "
            "from a single assumption stress. "
            "Blue bars increase the liability. Red bars decrease it."
        )

        sens_sorted = sens_df.sort_values("ICL Change", ascending=True)

        bar_colours = [
            COLOURS["icl"] if v > 0 else COLOURS["bel"]
            for v in sens_sorted["ICL Change"]
        ]

        fig_tornado = go.Figure(go.Bar(
            x            = sens_sorted["ICL Change"] / 1e6,
            y            = sens_sorted["Scenario"],
            orientation  = "h",
            marker_color = bar_colours,
            text         = [
                f"R{v/1e6:+.2f}M ({pct:+.1f}%)"
                for v, pct in zip(
                    sens_sorted["ICL Change"],
                    sens_sorted["ICL Change %"]
                )
            ],
            textposition = "outside",
            hovertemplate = (
                "<b>%{y}</b><br>"
                "ICL Change: R%{x:.2f}M<extra></extra>"
            ),
        ))
        fig_tornado.add_vline(x=0, line_color="#546e7a", line_width=1.5)
        fig_tornado.update_layout(
            height=420, template="plotly_white",
            xaxis_title="Change in ICL (R millions)",
            xaxis=dict(zeroline=True, zerolinecolor="#546e7a"),
            margin=dict(l=180, r=120),
        )
        st.plotly_chart(fig_tornado, use_container_width=True)

        # ── Sensitivity results table ──────────────────────────────────
        st.markdown("##### Sensitivity Results — Detailed Table")

        display_sens = sens_df[[
            "Scenario","Description","Stressed ICL","ICL Change","ICL Change %"
        ]].copy()
        display_sens["Base ICL"]    = base_icl
        display_sens["Stressed ICL"] = display_sens["Stressed ICL"].apply(
            lambda x: f"R{x:,.0f}"
        )
        display_sens["Base ICL"]    = f"R{base_icl:,.0f}"
        display_sens["ICL Change"]  = display_sens["ICL Change"].apply(
            lambda x: f"R{x:+,.0f}"
        )
        display_sens["ICL Change %"] = display_sens["ICL Change %"].apply(
            lambda x: f"{x:+.2f}%"
        )
        display_sens = display_sens[[
            "Scenario","Description","Base ICL",
            "Stressed ICL","ICL Change","ICL Change %"
        ]]

        st.dataframe(display_sens, use_container_width=True, hide_index=True)

        # ── Assumption interpretation ──────────────────────────────────
        st.markdown("##### Actuary's Commentary on Key Sensitivities")

        most_sensitive = sens_df.reindex(
            sens_df["ICL Change"].abs().sort_values(ascending=False).index
        ).iloc[0]
        least_sensitive = sens_df.reindex(
            sens_df["ICL Change"].abs().sort_values().index
        ).iloc[0]

        st.info(f"""
**Most Sensitive Assumption:** {most_sensitive['Scenario']}
→ A change in this assumption moves the total ICL by
R{most_sensitive['ICL Change']/1e6:.2f}M
({most_sensitive['ICL Change %']:+.1f}%).
This warrants close monitoring and robust experience studies.

**Least Sensitive Assumption:** {least_sensitive['Scenario']}
→ ICL impact of only R{least_sensitive['ICL Change']/1e6:.2f}M
({least_sensitive['ICL Change %']:+.1f}%),
indicating relative stability to this assumption.

**Discount Rate Sensitivity:** A 100 basis point movement in the
discount rate reflects the duration of the liability portfolio. Longer
duration books (whole life, long-term endowments) will show higher
interest rate sensitivity than short-duration term assurance books.

**Lapse Risk:** Where lapses reduce the ICL, the portfolio contains
contracts where future premiums are more valuable than the benefit
obligation — a typical pattern for longer-term endowment business.
Where lapses increase the ICL, the book contains contracts where the
insurer is better off if policies persist.
        """)

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════
        # REPORT FOOTER
        # ══════════════════════════════════════════════════════════════
        st.markdown(f"""
        <div style="
            background:#f5f5f5; border:1px solid #e0e0e0;
            padding:1.2rem 1.5rem; border-radius:8px;
            margin-top:1rem; color:#546e7a; font-size:0.85rem;
        ">
            <strong>Actuarial Report Certification</strong><br><br>
            This valuation report has been produced by the IFRS 17 / SAM
            Actuarial Engine as at <strong>{valuation_date_str}</strong>.
            The results are based on best estimate assumptions in accordance
            with IFRS 17 (General Measurement Model) and the South African
            Solvency Assessment and Management (SAM) framework.<br><br>
            <strong>Assumption Basis:</strong> {assumptions.name}<br>
            <strong>Discount Rate:</strong> {(flat_rate + illiquidity):.2%} per annum (flat rate approximation)<br>
            <strong>Risk Adjustment Confidence Level:</strong> {ra_confidence:.0%}<br>
            <strong>Mortality Table:</strong> SA85-90 Assured Lives (parametric approximation)<br>
            <strong>Products Valued:</strong> Term Assurance, Whole Life, Endowment (GMM)<br><br>
            <em>This report is generated for demonstration and educational purposes.
            It should not be used as a substitute for a signed actuarial opinion.</em>
        </div>
        """, unsafe_allow_html=True)

        # ── Download full report as CSV ────────────────────────────────
        st.markdown("&nbsp;")
        report_download = sens_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Sensitivity Analysis (CSV)",
            data      = report_download,
            file_name = f"sensitivity_analysis_{valuation_date_str}.csv",
            mime      = "text/csv",
            use_container_width=True,
        )
