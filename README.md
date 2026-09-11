# Clinical Policy & Risk Engine (CPRE)

### An End-to-End AI & Big Data Pipeline for Managed Care (HMO) Operations

CPRE is a demonstration pipeline that automates a real managed-care workflow: **prior-authorization risk evaluation and clinical policy retrieval**. Given a member ID, a procedure/diagnosis code, and free-text clinical chart notes, the system scores the member's clinical risk, profiles their social vulnerability, retrieves the governing Medicare coverage policy, checks the submitted notes against that policy's criteria, and issues an automated determination (expedited approval or a request for additional clinical information) — all in a single agentic pipeline built with LangGraph.

The project touches several distinct engineering disciplines end to end: big-data feature engineering (PySpark), supervised and unsupervised machine learning (PyTorch, XGBoost, scikit-learn), retrieval-augmented generation (LangChain + ChromaDB), and deterministic agentic orchestration (LangGraph). A Streamlit app ties it all together into an interactive demo.

**Live demo:** deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) — see [Deployment](#deployment-streamlit-community-cloud) below.

---

## Architecture

```
[ CMS Claims (SynPUF) + CDC SVI ] ──► [ PySpark ETL ] ──► [ Feature Store (Parquet) ]
                                                                     │
            ┌────────────────────────────────────────────────────────┴─────────────────────────────────┐
            ▼                                                                                          ▼
 [ PyTorch Tabular Neural Net + XGBoost ]                                                     [ ChromaDB Policy RAG ]
 - Predicts Prolonged Length-of-Stay Probability                                              - Indexes NCD/LCD Medicare Policies
 - k-Means / k-NN SDoH Vulnerability Clustering                                                - Local Sentence-Transformer Embeddings
            │                                                                                          │
            └────────────────────────────────► [ LangGraph Agent Orchestrator ] ◄───────────────────────┘
                                                              │
                                            [ Output: Expedited Approval or Clinical RFI ]
```

The agent itself is a 3-node deterministic `StateGraph` (`rag_agent/agent_orchestrator.py`):

1. **`member_risk_lookup`** — scores the member's inpatient risk with the PyTorch MLP and assigns a Social Determinants of Health (SDoH) vulnerability tier via k-Means clustering on CDC SVI percentiles.
2. **`policy_retrieval`** — semantically searches a ChromaDB vector store of Medicare LCD/NCD policy documents for the criteria relevant to the requested CPT code.
3. **`clinical_evaluator`** — checks the submitted chart notes against the retrieved policy's documented requirements (e.g. six weeks of conservative therapy, documented radiculopathy) and issues a decision with a cited rationale.

## What this project demonstrates

| Discipline | Where |
|---|---|
| **Big Data / ETL** | `etl/spark_feature_pipeline.py` — PySpark job that ingests CMS SynPUF beneficiary + inpatient + outpatient claims, computes longitudinal utilization features, joins CDC SVI county-level social vulnerability data, and writes a modeling-ready Parquet feature store. |
| **Supervised ML** | `models/train_models.py` — a custom PyTorch tabular MLP (`BatchNorm` + `Dropout`, trained with class-weighted `BCEWithLogitsLoss`) predicting prolonged inpatient stays, benchmarked against an XGBoost classifier. |
| **Unsupervised ML** | Same file — k-Means clustering of members into four SDoH vulnerability personas from CDC SVI percentiles, plus a k-NN index for peer-cohort lookups. |
| **RAG** | `rag_agent/policy_rag.py` — chunks and embeds Medicare coverage policy PDFs with a local, CPU-only sentence-transformer (`all-MiniLM-L6-v2`, no external API key required) into a persistent ChromaDB store, with clinical-criteria-aware chunk boundaries. |
| **Agentic workflows** | `rag_agent/agent_orchestrator.py` — LangGraph `StateGraph` chaining the risk model, the RAG retriever, and a rules-based clinical evaluator into one auditable pipeline. |
| **Interactive demo** | `app/main.py` — a Streamlit dashboard for running scenarios end to end and inspecting the model benchmarks, member SDoH profile, and cited policy grounding. |

**A note on scope, in the interest of accuracy:** the "clinical evaluator" node is currently a deterministic, keyword-based rules engine rather than an LLM call — it's what makes every decision in this demo fully explainable and reproducible. There is likewise no LLM fine-tuning step in the current codebase; the "custom" model here is the PyTorch MLP trained from scratch in `models/train_models.py`, not a fine-tuned pretrained model. If you're describing this project elsewhere (résumé, portfolio site), it's most accurate to say it demonstrates *custom neural network training* and *deterministic agentic orchestration* rather than LLM fine-tuning.

## Repository structure

```
clinical-policy-risk-engine/
├── app/
│   ├── main.py                     # Streamlit demo app
│   └── assets/cpretool.png         # UI logo
├── etl/
│   └── spark_feature_pipeline.py   # PySpark: raw claims + SVI -> feature store
├── models/
│   ├── train_models.py             # PyTorch MLP + XGBoost + k-Means/k-NN training
│   └── artifacts/                  # Trained model weights & preprocessors (committed)
│       ├── feature_names.pkl
│       ├── feature_scaler.pkl
│       ├── sdoh_scaler.pkl
│       ├── kmeans_sdoh.pkl
│       ├── knn_sdoh.pkl
│       ├── xgboost_model.pkl
│       └── pytorch_risk_mlp.pt
├── rag_agent/
│   ├── policy_rag.py               # ChromaDB ingestion & retrieval engine
│   └── agent_orchestrator.py       # LangGraph agent + PyTorch inference
├── data/
│   ├── raw/
│   │   ├── synpuf/                 # NOT in git -- see "Getting the data" below
│   │   ├── policies/                # CMS LCD/NCD policy PDFs (committed)
│   │   └── svi/                     # CDC SVI 2022 county file (committed)
│   └── processed/
│       ├── feature_store.parquet/   # PySpark output (committed, required at runtime)
│       └── chroma_policy_db/        # Pre-built vector index (committed, required at runtime)
├── requirements.txt                 # App + model runtime deps (used by Streamlit Cloud)
├── requirements-pipeline.txt        # Extra deps to re-run the PySpark ETL locally
└── .gitignore
```

### Why pre-built artifacts are committed

`models/artifacts/`, `data/processed/feature_store.parquet/`, and `data/processed/chroma_policy_db/` are technically derived, regenerable outputs — but they're committed to this repo on purpose. Streamlit Community Cloud only runs `app/main.py`; it does not run the PySpark ETL job or the RAG ingestion script on deploy. `ClinicalAgentPipeline.__init__` reads the feature store and model artifacts directly from disk, and `PolicyRAGEngine.load_existing_index()` raises `FileNotFoundError` if the Chroma store doesn't already exist. Without these files checked in, the deployed demo would not start.

The total size of everything committed (artifacts + processed data + policy PDFs + SVI file + source) is roughly 20MB — small enough for a fast clone and a fast Streamlit Cloud deploy.

## Getting the data

- **CMS DE-SynPUF (2008-2010 sample 1)** — synthetic Medicare beneficiary, inpatient, and outpatient claims. Not included in this repo (the outpatient claims file alone is ~162MB, over GitHub's 100MB per-file limit). Download the Beneficiary Summary and Inpatient/Outpatient Claims (Sample 1) files from CMS and place them under `data/raw/synpuf/` if you want to re-run the ETL pipeline:
  https://www.cms.gov/data-research/statistics-trends-reports/medicare-claims-synthetic-public-use-files/de-synpuf
- **CDC/ATSDR Social Vulnerability Index (2022, county-level)** — already included at `data/raw/svi/SVI_2022_US_COUNTY.csv`.
- **Medicare LCD/NCD policy documents** — three public CMS coverage determination PDFs are already included at `data/raw/policies/` (Lumbar MRI, Continuous Glucose Monitors, Total Knee Arthroplasty).

## Running locally

```bash
git clone https://github.com/almanuel7/clinical-policy-risk-engine.git
cd clinical-policy-risk-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app/main.py
```

The app loads the committed model artifacts and vector store directly, so no training or ETL step is required to try the demo.

## Regenerating the pipeline from scratch (optional)

Only needed if you want to rebuild the feature store, retrain the models, or re-ingest policy documents rather than use the committed artifacts.

```bash
pip install -r requirements.txt -r requirements-pipeline.txt   # adds PySpark (needs a local Java 17/11/8 runtime)

python etl/spark_feature_pipeline.py     # raw claims + SVI -> data/processed/feature_store.parquet
python models/train_models.py            # trains PyTorch MLP, XGBoost, k-Means/k-NN -> models/artifacts/
python rag_agent/policy_rag.py           # builds data/processed/chroma_policy_db/ from data/raw/policies/
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (public repos deploy for free on Community Cloud).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repository, branch `main`, and main file path `app/main.py`.
3. Streamlit Cloud installs from the root `requirements.txt` automatically — no extra configuration needed. It intentionally excludes PySpark (see `requirements-pipeline.txt` above) to keep the build fast; the deployed app only reads the already-committed feature store and model artifacts.
4. `requirements.txt` pins the CPU-only PyTorch wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`) so the build doesn't pull a much larger CUDA build that can exceed Community Cloud's build resource limits.
5. No secrets or API keys are required for this app — all embeddings run locally via `sentence-transformers`, and the clinical evaluator is rules-based.

## Model performance (from the last local training run)

| Model | Test ROC-AUC | Brier Score |
|---|---|---|
| PyTorch Tabular MLP (3-layer) | 0.824 | 0.089 |
| XGBoost Classifier (baseline) | 0.831 | — |

SDoH k-Means clustering (k=4) groups members into: Stable/Low Deprivation, Moderate Vulnerability, High Housing/Transportation Strain, and Severe Socioeconomic Disadvantage cohorts, driven by CDC SVI percentile ranks. Exact metrics will vary slightly between individual retrains due to random train/test splitting.

## Disclaimer

All data is synthetic (CMS DE-SynPUF) or public reference material (CDC SVI, published Medicare coverage policies). This project is a technical demonstration only and is not a certified utilization-management tool; the clinical logic is simplified for illustrative purposes and should not be used for actual coverage determinations.

---

Built by [@almanuel7](https://github.com/almanuel7)
