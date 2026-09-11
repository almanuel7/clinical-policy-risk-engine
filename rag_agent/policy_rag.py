"""
Clinical Policy & Risk Engine (CPRE)
Module: Policy RAG & ChromaDB Vector Store Engine
Description:
    1. Ingests Medicare/Medicaid clinical policy documents (PDF, TXT, MD).
    2. Chunks text using hierarchical boundary awareness to preserve clinical criteria.
    3. Generates dense embeddings locally using sentence-transformers (all-MiniLM-L6-v2).
    4. Indexes into a persistent ChromaDB vector store with rich metadata filtering.
    5. Provides a standardized query interface for policy citation and prior-auth verification.
"""

import os
import glob
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class PolicyRAGEngine:
    def __init__(
        self,
        policies_dir: str = "data/raw/policies",
        persist_dir: str = "data/processed/chroma_policy_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.policies_dir = policies_dir
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model

        # Initialize local HuggingFace embeddings running on CPU / Apple Silicon MPS
        print(f"Loading embedding model: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store: Optional[Chroma] = None

    def load_documents(self) -> List[Document]:
        """Loads all PDF, TXT, and Markdown documents from the policies directory."""
        supported_extensions = ["*.pdf", "*.txt", "*.md"]
        raw_files = []
        for ext in supported_extensions:
            raw_files.extend(glob.glob(os.path.join(self.policies_dir, ext)))

        if not raw_files:
            print(f"Warning: No policy files found in '{self.policies_dir}'.")
            return []

        print(f"Discovered {len(raw_files)} policy documents for ingestion.")
        documents: List[Document] = []

        for file_path in raw_files:
            filename = os.path.basename(file_path)
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                    loaded_docs = loader.load()

                # Enrich metadata with document level identifiers
                for doc in loaded_docs:
                    doc.metadata["file_name"] = filename
                    doc.metadata["policy_type"] = (
                        "NCD" if "ncd" in filename.lower()
                        else "LCD" if "lcd" in filename.lower()
                        else "COMMERCIAL_GUIDELINE"
                    )
                documents.extend(loaded_docs)
                print(f"  [✓] Loaded: {filename} ({len(loaded_docs)} pages/sections)")
            except Exception as e:
                print(f"  [✗] Failed loading {filename}: {str(e)}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into overlapping chunks using medical-policy-specific separators.
        Preserves complete paragraphs, list numbers, and clinical inclusion/exclusion blocks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=[
                "\n\n",
                "\nPOLICY:",
                "\nCOVERAGE CRITERIA:",
                "\nINDICATIONS AND LIMITATIONS:",
                "\nEXCLUSIONS:",
                "\n1. ",
                "\n2. ",
                "\n3. ",
                "\n- ",
                "\n",
                " ",
                ""
            ],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Generated {len(chunks)} contextual chunks across all policies.")
        return chunks

    def build_and_persist_index(self):
        """Executes the full ingestion, chunking, embedding, and indexing workflow."""
        docs = self.load_documents()
        if not docs:
            print("Aborting index build: No documents available.")
            return

        chunks = self.chunk_documents(docs)

        os.makedirs(self.persist_dir, exist_ok=True)
        print(f"\nBuilding persistent ChromaDB vector store at: {self.persist_dir}...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="medicare_medicaid_policies",
        )
        print(">> Vector index successfully constructed and saved to disk.")

    def load_existing_index(self):
        """Loads a pre-built ChromaDB collection from local disk."""
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(f"ChromaDB store not found at: {self.persist_dir}. Run build first.")
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="medicare_medicaid_policies",
        )
        print(f">> Loaded existing ChromaDB index from {self.persist_dir}")

    def query_policy(
        self,
        query: str,
        k: int = 3,
        policy_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the policy index for semantic relevance and returns formatted citations.
        """
        if self.vector_store is None:
            self.load_existing_index()

        filter_dict = {"policy_type": policy_type_filter} if policy_type_filter else None

        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_dict,
        )

        formatted_results = []
        for doc, score in results:
            page_num = doc.metadata.get("page", doc.metadata.get("page_number", 1))
            formatted_results.append({
                "file_name": doc.metadata.get("file_name", "Unknown Policy"),
                "policy_type": doc.metadata.get("policy_type", "Standard"),
                "page": page_num,
                "relevance_score": round(float(score), 4),
                "content": doc.page_content.strip()
            })

        return formatted_results


# ---------------------------------------------------------
# CLI Execution & Standalone Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    POLICIES_FOLDER = os.path.join(BASE_DIR, "data/raw/policies")
    CHROMA_FOLDER = os.path.join(BASE_DIR, "data/processed/chroma_policy_db")

    # If the user has no policy documents yet, create sample policy text files
    if not os.path.exists(POLICIES_FOLDER) or not os.listdir(POLICIES_FOLDER):
        os.makedirs(POLICIES_FOLDER, exist_ok=True)
        sample_lumbar_mri = """
POLICY: Lumbar Spine MRI (CPT Codes: 72148, 72149, 72158)
POLICY TYPE: Local Coverage Determination (LCD)
EFFECTIVE DATE: 2024-01-01

COVERAGE CRITERIA:
Lumbar Spine Magnetic Resonance Imaging (MRI) is considered medically necessary only when ONE of the following clinical conditions is documented in the medical record:
1. Acute Neurological Emergency: Severe or rapidly progressive motor deficit, suspected cauda equina syndrome (e.g., saddle anesthesia, bowel/bladder dysfunction), or spinal cord compression.
2. Failure of Conservative Therapy: Persistent, severe low back pain with radiculopathy that has failed at least six (6) weeks of documented, active conservative management. Conservative management MUST include:
   - Supervised physical therapy (minimum 6 visits) or active home exercise program.
   - Pharmacotherapy trial (e.g., NSAIDs, acetaminophen, or neuropathic agents, unless contraindicated).
3. Red Flag Symptoms: Suspected spinal malignancy, severe acute trauma, or active spinal infection (e.g., osteomyelitis/discitis) accompanied by fever or elevated inflammatory markers (ESR/CRP).

NON-COVERED / EXCLUSIONS:
- Uncomplicated acute low back pain without red flags or radicular symptoms during the initial 6-week period is NOT covered.
- Routine repeat imaging without documented clinical change or new acute trauma is not medically necessary.
"""
        with open(os.path.join(POLICIES_FOLDER, "LCD_Lumbar_MRI_Guidelines.txt"), "w") as f:
            f.write(sample_lumbar_mri.strip())

        sample_cgm = """
POLICY: Continuous Glucose Monitoring (CGM) Systems (CPT: 95249, 95250, 95251, HCPCS: E2103)
POLICY TYPE: National Coverage Determination (NCD)

COVERAGE CRITERIA:
CGM devices and related supplies are covered for patients who meet ALL of the following criteria:
1. Beneficiary has a confirmed diagnosis of diabetes mellitus (Type 1 or Type 2).
2. Beneficiary is currently being treated with insulin or has a documented history of severe, recurring hypoglycemia.
3. The treating healthcare provider has determined that the patient (or caregiver) has completed comprehensive diabetes self-management training and can safely adjust insulin based on CGM trends.
4. The patient has a follow-up assessment scheduled within 6 months of initiating CGM to evaluate ongoing compliance and clinical efficacy (e.g., HbA1c trajectory).
"""
        with open(os.path.join(POLICIES_FOLDER, "NCD_Continuous_Glucose_Monitors.txt"), "w") as f:
            f.write(sample_cgm.strip())
        print(f"Generated default clinical policy templates in {POLICIES_FOLDER}/")

    # Run Ingestion & Index Construction
    rag = PolicyRAGEngine(
        policies_dir=POLICIES_FOLDER,
        persist_dir=CHROMA_FOLDER,
    )
    rag.build_and_persist_index()

    # Run Test Query
    print("\n--- Running Sample Clinical Prior-Auth Query ---")
    test_query = "What are the conservative therapy requirements before approving an MRI for lumbar spine radiculopathy?"
    citations = rag.query_policy(test_query, k=2)

    for i, cite in enumerate(citations, 1):
        print(f"\n[Citation {i}] - Source: {cite['file_name']} (Page {cite['page']}) | Score: {cite['relevance_score']}")
        print(f"Content:\n{cite['content']}\n" + "-" * 50)