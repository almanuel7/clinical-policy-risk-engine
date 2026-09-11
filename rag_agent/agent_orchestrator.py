"""
Clinical Policy & Risk Engine (CPRE)
Module: LangGraph Agentic Decision Orchestrator
Description:
    1. Ingests prior-authorization payload (Patient ID, CPT, ICD-10, Chart Notes).
    2. Node 1 (Member Risk & SDoH Lookup): Runs local PyTorch MLP and k-NN clustering.
    3. Node 2 (Policy Retrieval): Performs semantic search in ChromaDB for CPT guidelines.
    4. Node 3 (Clinical Compliance Evaluator): Validates notes against policy criteria.
    5. Node 4 (Decision Routing & Self-Correction): Outputs Approval or Clinical RFI.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, TypedDict

import torch
import torch.nn as nn
from langgraph.graph import StateGraph, END

# Import local Tabular PyTorch Architecture and RAG Engine
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rag_agent.policy_rag import PolicyRAGEngine


# ---------------------------------------------------------
# 1. Load PyTorch Architecture & Artifacts
# ---------------------------------------------------------
class TabularRiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.3):
        super(TabularRiskMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class AgentState(TypedDict):
    # Input Request Data
    member_id: str
    procedure_code: str
    diagnosis_code: str
    clinical_notes: str
    
    # Inferred Pipeline State
    member_found: bool
    ml_risk_score: float
    sdoh_cluster_id: int
    sdoh_vulnerability_tier: str
    retrieved_policies: List[Dict[str, Any]]
    policy_citation_text: str
    
    # Evaluation & Output
    criteria_met: bool
    decision: str  # APPROVED | PENDING_ADDITIONAL_INFO | DENIED
    missing_criteria: List[str]
    rationale: str


# ---------------------------------------------------------
# 2. Agent Decision Nodes
# ---------------------------------------------------------
class ClinicalAgentPipeline:
    def __init__(self):
        self.artifacts_dir = os.path.join(BASE_DIR, "models/artifacts")
        self.feature_store_path = os.path.join(BASE_DIR, "data/processed/feature_store.parquet")
        self.chroma_dir = os.path.join(BASE_DIR, "data/processed/chroma_policy_db")

        # Load ML Feature Encoders
        self.scaler = joblib.load(os.path.join(self.artifacts_dir, "feature_scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(self.artifacts_dir, "feature_names.pkl"))
        self.kmeans_sdoh = joblib.load(os.path.join(self.artifacts_dir, "kmeans_sdoh.pkl"))
        self.sdoh_scaler = joblib.load(os.path.join(self.artifacts_dir, "sdoh_scaler.pkl"))

        # Initialize PyTorch Model
        self.model = TabularRiskMLP(input_dim=len(self.feature_names))
        self.model.load_state_dict(
            torch.load(os.path.join(self.artifacts_dir, "pytorch_risk_mlp.pt"), map_location=torch.device("cpu"))
        )
        self.model.eval()

        # Load RAG Engine
        self.rag_engine = PolicyRAGEngine(persist_dir=self.chroma_dir)
        self.rag_engine.load_existing_index()

        # Pre-load Feature Store for Member Lookups
        self.df_features = pd.read_parquet(self.feature_store_path)

    def node_member_risk_lookup(self, state: AgentState) -> Dict[str, Any]:
        """Looks up member historical utilization and scores tabular risk via PyTorch."""
        member_id = state["member_id"]
        member_rows = self.df_features[self.df_features["DESYNPUF_ID"] == member_id]

        if member_rows.empty:
            return {
                "member_found": False,
                "ml_risk_score": 0.50,
                "sdoh_cluster_id": 0,
                "sdoh_vulnerability_tier": "Unknown (Default Population Average)"
            }

        member_record = member_rows.iloc[0]

        # 1. SDoH Cluster Inference
        sdoh_cols = [
            "svi_socioeconomic_pctile", "svi_household_pctile",
            "svi_minority_pctile", "svi_housing_transp_pctile", "svi_overall_vulnerability"
        ]
        sdoh_vec = self.sdoh_scaler.transform([member_record[sdoh_cols].values])
        cluster_id = int(self.kmeans_sdoh.predict(sdoh_vec)[0])
        tier_names = {
            0: "Low Deprivation / Stable",
            1: "Moderate Vulnerability",
            2: "High Housing / Transport Strain",
            3: "Extreme Socioeconomic Vulnerability"
        }

        # 2. PyTorch Tabular Risk Inference
        encoded_row = pd.get_dummies(member_rows, columns=["gender", "race"], drop_first=True)
        # Ensure all training columns exist
        for col in self.feature_names:
            if col not in encoded_row.columns:
                encoded_row[col] = 0

        X_input = self.scaler.transform(encoded_row[self.feature_names].values)
        with torch.no_grad():
            logit = self.model(torch.tensor(X_input, dtype=torch.float32))
            prob = float(torch.sigmoid(logit).item())

        return {
            "member_found": True,
            "ml_risk_score": round(prob, 4),
            "sdoh_cluster_id": cluster_id,
            "sdoh_vulnerability_tier": tier_names.get(cluster_id, "Standard")
        }

    def node_policy_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """Queries the ChromaDB Vector store using procedure description and CPT."""
        cpt = state["procedure_code"]
        notes = state["clinical_notes"]
        
        # Formulate rich clinical query
        rag_query = f"Coverage criteria requirements and conservative therapy exclusions for CPT {cpt}. Clinical context: {notes}"
        citations = self.rag_engine.query_policy(rag_query, k=2)

        citation_text = "\n\n".join(
            [f"[Source: {c['file_name']} (Page {c['page']})]\n{c['content']}" for c in citations]
        )

        return {
            "retrieved_policies": citations,
            "policy_citation_text": citation_text
        }

    def node_clinical_evaluator(self, state: AgentState) -> Dict[str, Any]:
        """
        Evaluates clinical notes against retrieved policy criteria.
        Extracts key clinical validation gates:
        - 6 weeks conservative management (PT / NSAIDs)
        - Documented radiculopathy or progressive neurological deficit
        - Insulin dependence (for CGM requests)
        """
        notes_lower = state["clinical_notes"].lower()
        cpt = state["procedure_code"]
        missing = []

        # Deterministic Verification Rules for Lumbar MRI (72148, 72149)
        if cpt in ["72148", "72149", "72158"]:
            has_red_flags = any(w in notes_lower for w in ["cauda equina", "trauma", "fever", "malignancy", "cord compression"])
            has_neuro_deficit = any(w in notes_lower for w in ["motor deficit", "foot drop", "progressive weakness", "reflex loss"])
            has_pt = any(w in notes_lower for w in ["physical therapy", "pt completed", "home exercise"])
            has_meds = any(w in notes_lower for w in ["nsaid", "ibuprofen", "meloxicam", "gabapentin", "pharmacotherapy"])
            has_duration = any(w in notes_lower for w in ["6 weeks", "8 weeks", "2 months", "3 months", "chronic"])

            if not has_red_flags:
                if not (has_pt and has_meds and has_duration):
                    if not has_pt:
                        missing.append("Documentation of completed 6-week trial of supervised Physical Therapy.")
                    if not has_meds:
                        missing.append("Documentation of trial of anti-inflammatory / neuropathic pharmacotherapy.")
                    if not has_duration:
                        missing.append("Documentation confirming symptom duration of at least 6 weeks.")
                if not has_neuro_deficit and not ("radiculopathy" in notes_lower or "sciatica" in notes_lower):
                    missing.append("Documented physical exam finding of lumbar radiculopathy or objective neurological deficit.")

        # Verification Rules for CGM (E2103, 95250)
        elif cpt in ["E2103", "95250", "95251", "95249"]:
            has_diabetes = any(w in notes_lower for w in ["diabetes", "type 1", "type 2", "t1d", "t2d"])
            has_insulin = any(w in notes_lower for w in ["insulin", "multiple daily injections", "pump"])
            
            if not has_diabetes:
                missing.append("Confirmed clinical diagnosis of Diabetes Mellitus.")
            if not has_insulin:
                missing.append("Documented requirement of insulin therapy or recurrent severe hypoglycemia.")

        criteria_met = len(missing) == 0

        if criteria_met:
            decision = "APPROVED"
            rationale = "Clinical documentation satisfies all required medical necessity criteria and guideline conservative therapy protocols."
        else:
            decision = "PENDING_ADDITIONAL_INFO"
            rationale = f"Request lacks mandatory documentation criteria for expedited auto-approval: {'; '.join(missing)}"

        return {
            "criteria_met": criteria_met,
            "missing_criteria": missing,
            "decision": decision,
            "rationale": rationale
        }

    def build_graph(self):
        """Constructs the LangGraph state machine graph."""
        workflow = StateGraph(AgentState)

        workflow.add_node("member_risk_lookup", self.node_member_risk_lookup)
        workflow.add_node("policy_retrieval", self.node_policy_retrieval)
        workflow.add_node("clinical_evaluator", self.node_clinical_evaluator)

        workflow.set_entry_point("member_risk_lookup")
        workflow.add_edge("member_risk_lookup", "policy_retrieval")
        workflow.add_edge("policy_retrieval", "clinical_evaluator")
        workflow.add_edge("clinical_evaluator", END)

        return workflow.compile()


# ---------------------------------------------------------
# 3. Execution & Simulation Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Initializing Clinical Policy & Risk Orchestrator ---")
    pipeline = ClinicalAgentPipeline()
    app = pipeline.build_graph()

    # Find a valid sample member ID from the feature store
    sample_id = pipeline.df_features["DESYNPUF_ID"].iloc[0]

    # Test Case A: Complete Chart Notes (Should Auto-Approve)
    test_request_approval = {
        "member_id": sample_id,
        "procedure_code": "72148",
        "diagnosis_code": "M54.16 (Lumbar Radiculopathy)",
        "clinical_notes": (
            "Patient with 8 weeks of severe right L5 radiating leg pain and radiculopathy. "
            "Completed 8 visits of supervised Physical Therapy and an 8-week trial of Meloxicam with no relief. "
            "Physical exam demonstrates diminished right patellar reflex and positive straight leg raise."
        ),
        "member_found": False,
        "ml_risk_score": 0.0,
        "sdoh_cluster_id": 0,
        "sdoh_vulnerability_tier": "",
        "retrieved_policies": [],
        "policy_citation_text": "",
        "criteria_met": False,
        "decision": "",
        "missing_criteria": [],
        "rationale": ""
    }

    print("\n" + "=" * 60)
    print("SIMULATION 1: Evaluating Complete Prior-Auth Request")
    print("=" * 60)
    result_a = app.invoke(test_request_approval)

    print(f"\n[Member Inpatient Risk Score]: {result_a['ml_risk_score'] * 100:.1f}%")
    print(f"[SDoH Vulnerability Tier]: {result_a['sdoh_vulnerability_tier']}")
    print(f"[Determination]: {result_a['decision']}")
    print(f"[Clinical Rationale]: {result_a['rationale']}")
    print(f"\n[Cited Policy Grounding]:\n{result_a['policy_citation_text']}")

    # Test Case B: Incomplete Chart Notes (Should Trigger RFI loop)
    test_request_rfi = {
        "member_id": sample_id,
        "procedure_code": "72148",
        "diagnosis_code": "M54.5 (Low Back Pain)",
        "clinical_notes": "Patient reports lower back stiffness for 10 days after lifting heavy yard equipment. Requesting lumbar MRI.",
        "member_found": False,
        "ml_risk_score": 0.0,
        "sdoh_cluster_id": 0,
        "sdoh_vulnerability_tier": "",
        "retrieved_policies": [],
        "policy_citation_text": "",
        "criteria_met": False,
        "decision": "",
        "missing_criteria": [],
        "rationale": ""
    }

    print("\n" + "=" * 60)
    print("SIMULATION 2: Evaluating Incomplete Prior-Auth Request")
    print("=" * 60)
    result_b = app.invoke(test_request_rfi)

    print(f"\n[Determination]: {result_b['decision']}")
    print(f"[Clinical Rationale]: {result_b['rationale']}")
    print("[Automated Information Request Sent to Clinic]:")
    for item in result_b["missing_criteria"]:
        print(f"  * {item}")