"""
Clinical Policy & Risk Engine (CPRE)
Interactive Demonstration & Executive Operations Dashboard
"""

import os
import sys
import pandas as pd
import streamlit as st
import base64

# Ensure repository root is on sys.path.
# NOTE: __file__ is app/main.py, so this must go up TWO directory levels
# (app/ -> repo root), not one -- otherwise "from rag_agent..." only
# resolves when something else (an IDE, `python -m`, a stray PYTHONPATH)
# happens to already have the repo root on sys.path, which is exactly why
# this worked locally but broke on Streamlit Community Cloud's bare
# `streamlit run app/main.py` invocation.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rag_agent.agent_orchestrator import ClinicalAgentPipeline

def get_image_base64(path: str):
    """Reads an image file from disk and returns a base64 data URI, or None if missing."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    return f"data:image/{ext};base64,{encoded}"


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Policy & Risk Engine (CPRE) v1",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .status-badge-approved { background-color: #DEF7EC; color: #03543F; padding: 6px 14px; border-radius: 6px; font-weight: 600; }
    .status-badge-rfi { background-color: #FEF3C7; color: #92400E; padding: 6px 14px; border-radius: 6px; font-weight: 600; }
    .citation-box { background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 12px; border-radius: 4px; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Pipeline Initialization (Cached for performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Neural Net & Vector Index...")
def load_orchestrator():
    pipeline = ClinicalAgentPipeline()
    graph_app = pipeline.build_graph()
    return pipeline, graph_app


try:
    pipeline, graph_app = load_orchestrator()
    df_features = pipeline.df_features
except Exception as e:
    st.error(f"Error loading system artifacts: {str(e)}")
    st.info(
        "Please ensure `spark_feature_pipeline.py`, `train_models.py`, and `policy_rag.py` have all been executed."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar: Member Selector & Clinical Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.title("CPRE Control Panel")
    st.caption("Multimodal Managed Care Decision Engine")
    st.divider()

    st.subheader("1. Member Selection")
    sample_members = df_features["DESYNPUF_ID"].head(50).tolist()
    selected_member_id = st.selectbox("Select Beneficiary ID:", sample_members, index=0)

    member_data = df_features[df_features["DESYNPUF_ID"] == selected_member_id].iloc[0]

    st.markdown("### 2. Prior-Auth Request")
    request_type = st.radio(
        "Preset Demonstration Scenario:",
        [
            "Lumbar Spine MRI (Meets Criteria - PT & Radiculopathy)",
            "Lumbar Spine MRI (Incomplete - Missing 6-Wk PT Trial)",
            "Continuous Glucose Monitor (Meets Criteria - Insulin)",
            "Custom Prior-Auth Entry",
        ],
    )

    if request_type == "Lumbar Spine MRI (Meets Criteria - PT & Radiculopathy)":
        default_cpt = "72148"
        default_icd = "M54.16 (Lumbar Radiculopathy)"
        default_notes = (
            "Patient presents with 8 weeks of severe right L5 radiating leg pain and numbness. "
            "Has completed 8 visits of supervised Physical Therapy and an 8-week trial of Meloxicam with no relief. "
            "Physical exam demonstrates positive straight leg raise and diminished right patellar reflex."
        )
    elif request_type == "Lumbar Spine MRI (Incomplete - Missing 6-Wk PT Trial)":
        default_cpt = "72148"
        default_icd = "M54.5 (Low Back Pain)"
        default_notes = (
            "Patient reports 10 days of acute lower back soreness after lifting boxes. "
            "No neurological deficits noted on exam. Requesting lumbar MRI to rule out disc herniation."
        )
    elif request_type == "Continuous Glucose Monitor (Meets Criteria - Insulin)":
        default_cpt = "E2103"
        default_icd = "E11.65 (Type 2 Diabetes with Hyperglycemia)"
        default_notes = (
            "Confirmed Type 2 Diabetes Mellitus managed with multiple daily insulin injections (Lantus + Novolog). "
            "Patient completed comprehensive diabetes education program and requires real-time CGM for frequent nocturnal hypoglycemia."
        )
    else:
        default_cpt = "72148"
        default_icd = "M54.16"
        default_notes = ""

    selected_cpt = st.text_input("Procedure / CPT Code:", default_cpt)
    selected_icd = st.text_input("Diagnosis / ICD-10 Code:", default_icd)
    clinical_notes = st.text_area("Submitted Clinical Chart Notes:", default_notes, height=140)

    run_evaluation = st.button("Run Autonomous Clinical Evaluation", type="primary", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Made in :streamlit: by [@antLmanuel7](https://antmanuel-portfolio.site/)")


# ---------------------------------------------------------
# Main Page Layout
# ---------------------------------------------------------

# Load and encode logo
logo_path = os.path.join(os.path.dirname(__file__), "assets", "cpretool.png")
logo_data_uri = get_image_base64(logo_path)

# Logo HTML element (falls back gracefully if missing)
logo_html = f'<img src="{logo_data_uri}" style="width: 52px; height: 52px; object-fit: contain; margin-right: 16px; border-radius: 8px;">' if logo_data_uri else ""

#Header
st.markdown(
    f"""
    <div class="main-header" style="color: #1E3A8A; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center;">
                {logo_html}
                <div>
                    <h2 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #1E3A8A; line-height: 1.2;">
                        Clinical Policy & Risk Engine (CPRE)
                    </h2>
                    <p class="sub-header" style="margin: 4px 0 0 0; font-size: 1.05rem; color: #4B5563;">
                        Demo of End-to-End Multimodal AI Pipeline Utilizing Clinical Policy & Risk Engine (CPRE) to bridge tabular predictive modeling with generative AI.
                    </p>
                </div>
            </div>
            <div style="text-align: right; font-size: 0.85rem; color: #4B5563; line-height: 1.4;">
                <span>Backend Pipeline: PySpark, PyTorch + SDoH Engine, ChromaDB Vector Store, LangGraph State Graph</span><br>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Member Age / Gender", value=f"{int(member_data['age'])} Yrs | {member_data['gender']}")
with kpi2:
    st.metric(label="Comorbidities", value=f"{int(member_data['comorbidity_count'])} Conditions")
with kpi3:
    st.metric(label="Inpatient Admits (12M)", value=f"{int(member_data['total_inpatient_admissions'])}")
with kpi4:
    st.metric(label="Total Inpatient Paid", value=f"${member_data['total_inpatient_paid']:,.2f}")

st.divider()

# Tabbed Layout
tab_agent, tab_member_profile, tab_model_architecture = st.tabs(
    ["Agent Decision & Policy Grounding", "Member 360 & SDoH Risk", "Architecture & ML Benchmarks"]
)

# ---------------------------------------------------------
# TAB 1: Agent Execution & RAG Grounding
# ---------------------------------------------------------
with tab_agent:
    if run_evaluation:
        with st.spinner("Executing LangGraph State Graph..."):
            initial_state = {
                "member_id": selected_member_id,
                "procedure_code": selected_cpt,
                "diagnosis_code": selected_icd,
                "clinical_notes": clinical_notes,
                "member_found": False,
                "ml_risk_score": 0.0,
                "sdoh_cluster_id": 0,
                "sdoh_vulnerability_tier": "",
                "retrieved_policies": [],
                "policy_citation_text": "",
                "criteria_met": False,
                "decision": "",
                "missing_criteria": [],
                "rationale": "",
            }

            output = graph_app.invoke(initial_state)

        col_dec, col_risk = st.columns([2, 1])

        with col_dec:
            st.subheader("Autonomous Clinical Determination")
            if output["decision"] == "APPROVED":
                st.markdown(
                    '<span class="status-badge-approved">✅ EXPEDITED APPROVAL ISSUED</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge-rfi">⚠️ PENDING ADDITIONAL CLINICAL INFORMATION (RFI)</span>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown(f"**Clinical Rationale:** {output['rationale']}")

            if output["missing_criteria"]:
                st.error("Actionable Information Required from Provider:")
                for item in output["missing_criteria"]:
                    st.markdown(f"- **{item}**")

        with col_risk:
            st.subheader("Predictive Risk Scores")
            risk_pct = output["ml_risk_score"] * 100
            st.metric(
                label="PyTorch Inpatient Risk Probability",
                value=f"{risk_pct:.1f}%",
                delta="High Risk" if risk_pct > 35 else "Low-Moderate Risk",
                delta_color="inverse",
            )
            st.markdown(f"**SDoH Vulnerability:** `{output['sdoh_vulnerability_tier']}`")

        st.divider()

        st.subheader("📚 Grounded Clinical Policy Citations (ChromaDB Retrieval)")
        if output["retrieved_policies"]:
            for idx, citation in enumerate(output["retrieved_policies"], 1):
                with st.expander(
                    f"Citation {idx}: {citation['file_name']} (Relevance Score: {citation['relevance_score']})",
                    expanded=True,
                ):
                    st.markdown(f'<div class="citation-box">{citation["content"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No explicit policy manual was found matching this procedure code.")
    else:
        st.info("👈 Select a scenario and click **Run Autonomous Clinical Evaluation** in the sidebar to execute the engine.")


# ---------------------------------------------------------
# TAB 2: Member 360 & SDoH Clustering
# ---------------------------------------------------------
with tab_member_profile:
    st.subheader(f"Member 360 Overview: {selected_member_id}")

    col_diag, col_sdoh = st.columns(2)

    with col_diag:
        st.markdown("#### Documented Chronic Conditions (CMS SynPUF)")
        conditions = {
            "Diabetes": member_data.get("has_diabetes", 0),
            "Congestive Heart Failure (CHF)": member_data.get("has_chf", 0),
            "Chronic Kidney Disease": member_data.get("has_chrnkidn", 0),
            "COPD": member_data.get("has_copd", 0),
            "Depression": member_data.get("has_depressn", 0),
            "Ischemic Heart Disease": member_data.get("has_ischmcht", 0),
            "Rheumatoid / Osteoarthritis": member_data.get("has_ra_oa", 0),
            "Stroke / TIA": member_data.get("has_strketia", 0),
        }
        cond_df = pd.DataFrame(
            [{"Condition": k, "Status": "Diagnosed" if v == 1 else "None"} for k, v in conditions.items()]
        )
        st.dataframe(cond_df, use_container_width=True, hide_index=True)

    with col_sdoh:
        st.markdown("#### CDC Social Vulnerability Index Percentiles (County FIPS)")
        sdoh_metrics = pd.DataFrame(
            {
                "SDoH Theme": [
                    "Socioeconomic Deprivation",
                    "Household Composition",
                    "Minority Status",
                    "Housing & Transportation",
                    "Overall Social Vulnerability",
                ],
                "Percentile Rank": [
                    member_data.get("svi_socioeconomic_pctile", 0.5),
                    member_data.get("svi_household_pctile", 0.5),
                    member_data.get("svi_minority_pctile", 0.5),
                    member_data.get("svi_housing_transp_pctile", 0.5),
                    member_data.get("svi_overall_vulnerability", 0.5),
                ],
            }
        )
        st.dataframe(
            sdoh_metrics.style.format({"Percentile Rank": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# TAB 3: Architecture & Model Benchmarks
# ---------------------------------------------------------
with tab_model_architecture:
    st.subheader("System Architecture & Benchmark Evaluation")

    st.markdown(
        """
        ```
        [ CMS Claims (SynPUF) + CDC SVI ] ──> [ PySpark ETL ] ──> [ Feature Store (Parquet) ]
                                                                             │
                    ┌────────────────────────────────────────────────────────┴─────────────────────────────────┐
                    ▼                                                                                          ▼
         [ PyTorch Tabular Neural Net ]                                                             [ ChromaDB Policy RAG ]
         - Predicts Prolonged LOS Probability                                                       - Indexes NCD/LCD Policies
         - Weighted BCE Loss for Imbalance                                                          - Dense Vector Similarity
                    │                                                                                          │
                    └────────────────────────────────► [ LangGraph Orchestrator ] ◄────────────────────────────┘
                                                              │
                                            [ Output: Expedited Approval or RFI ]
        ```
        """
    )

    st.divider()

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("#### Tabular Model Performance Benchmark")
        benchmarks = pd.DataFrame(
            {
                "Model Architecture": ["PyTorch Tabular MLP (3-Layer)", "XGBoost Classifier Baseline"],
                "Test ROC-AUC": ["0.824", "0.831"],
                "Brier Loss Score": ["0.089", "0.084"],
                "Latency (Inference)": ["12 ms", "8 ms"],
            }
        )
        st.dataframe(benchmarks, use_container_width=True, hide_index=True)

    with m_col2:
        st.markdown("#### Unsupervised SDoH Cluster Profiles ($k$-Means)")
        clusters = pd.DataFrame(
            {
                "Cluster ID": [0, 1, 2, 3],
                "Cohort Name": [
                    "Stable / Low Deprivation",
                    "Moderate Vulnerability",
                    "High Housing / Transportation Strain",
                    "Severe Socioeconomic Disadvantage",
                ],
                "Population Share": ["38%", "27%", "21%", "14%"],
            }
        )
        st.dataframe(clusters, use_container_width=True, hide_index=True)