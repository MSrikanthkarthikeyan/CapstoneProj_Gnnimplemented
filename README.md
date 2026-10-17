# PathBuilder: Agentic AI Learning Path Builder & GNN Recommendation Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![PyG](https://img.shields.io/badge/PyG-GCN%2FGAT-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![Gemini 2.5](https://img.shields.io/badge/Gemini-2.5%20Flash-blueviolet.svg)](https://deepmind.google/technologies/gemini/)

> **Research Question**: *Can graph neural networks improve prerequisite-aware, personalized learning-path recommendation compared with conventional resource ranking and graph-based baselines?*

For full mathematical formulations, detailed tech stack, workflow diagrams, and scientific evaluation benchmarks, see **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**.

---

## ⚡ Quick Start

### 1. Configure Environment
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
BACKEND_URL=http://localhost:8000
```

### 2. Launch FastAPI Backend
```bash
.\venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --port 8000
```
* **Swagger Docs**: `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`

### 3. Launch Streamlit Interactive UI
```bash
.\venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py --server.port 8501
```
* Access the interactive learner UI at `http://localhost:8501`.
* Includes 3 integrated tabs: **Interactive Journey & Tutor**, **GNN Recommendations**, and **Evaluation Portal**.

### 4. Launch Research Evaluation Dashboard
```bash
.\venv\Scripts\python.exe -m streamlit run evaluation/dashboard.py --server.port 8502
```
* Access the evaluation dashboard at `http://localhost:8502`.

### 5. Generate Journal Paper Results & Figures (300 DPI)
```bash
.\venv\Scripts\python.exe generate_paper_results.py
```
* Generates multi-seed publication benchmarks across 3 curricula.
* Outputs 5 high-resolution figures (`fig1_training_loss.png` through `fig5_tsne_embeddings.png`) and ready-to-paste LaTeX tables in `paper_artifacts/JOURNAL_RESULTS.md`.

### 6. Run Automated Test Suites
```bash
# End-to-end GNN architecture, features, scoring, baselines, and evaluation
.\venv\Scripts\python.exe -m unittest test_gnn_pipeline.py -v

# FastAPI REST endpoint integration tests
.\venv\Scripts\python.exe -m unittest test_api_endpoints.py -v
```

---

## 📐 Architecture Highlights

* **Multi-Source Ingestion (Phase A)**: Automatically retrieves curriculum context from Wikipedia, YouTube tutorials, and MIT OpenCourseWare.
* **Prerequisite DAG Synthesis (Phase B)**: NetworkX-validated acyclic knowledge graphs generated via Gemini 2.5.
* **ML Resource Ranking & MMR Diversification**: Extracts 9-dimensional features (Okapi BM25, TF-IDF n-gram similarity, difficulty match, source authority, duration fit, freshness, popularity), computes Ensemble Scoring, and executes Maximal Marginal Relevance (MMR) Diversification ($\lambda=0.65$) to bind Top-3 diverse resources (video, article, course) to each concept.
* **GNN Representation Learning (10-Dim Fusion)**: 2-layer Graph Convolutional Network (GCN) and Graph Attention Network (GAT) trained via pairwise BPR loss over node feature vectors fusing graph topology, learner mastery state, and ML resource signals (BM25, TF-IDF, Resource Authority).
* **Composite Recommendation Scoring**: Weighted scoring combining Prerequisite Readiness ($\alpha=0.40$), Target Relevance ($\beta=0.35$), and Marginal Information Gain ($\gamma=0.25$).
* **Strict Prerequisite Constraint**: Zero prerequisite violations ($PVR = 0.00$) via dynamic topological barrier filtering.
* **Adaptive Tutor Loop (Phase C)**: LangGraph state machine delivering micro-lessons grounded in the curated Top-3 resources and grading quizzes with real-time graph unlocks.
* **Scientific Evaluation Harness**: Automated comparative evaluation across NDCG@5, NDCG@10, MRR, PVR, and ablation models.

Detailed specifications and documentation are maintained in **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**.
