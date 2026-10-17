# PathBuilder: Agentic AI Learning Path Builder with GNN Recommendation Engine

> **Research Focus**: *Can graph neural networks improve prerequisite-aware, personalized learning-path recommendation compared with conventional resource ranking and graph-based baselines?*

---

## 1. Executive Summary & Context

**PathBuilder** is an agentic, graph-based personalized learning platform designed to solve a fundamental deficiency in online self-directed education: linear, one-size-fits-all curricula that ignore prerequisite dependencies and dynamic learner mastery.

When a learner provides an arbitrary high-level goal (e.g., *"Quantum Computing"*, *"Distributed Consensus"*, *"Graph Neural Networks"*), PathBuilder executes a 6-phase pipeline:
1. **Multi-Source Ingestion (Phase A)**: Pulls real-world educational material across Wikipedia, YouTube tutorials, and MIT OpenCourseWare (Articles, Videos, Syllabi, Course Notes).
2. **Prerequisite DAG Synthesis (Phase B)**: Prompts Gemini 2.5 with curriculum constraints to generate a formal Directed Acyclic Graph (DAG) validated for acyclicity and connectivity via NetworkX.
3. **ML Resource Ranking & MMR Diversification**:
   - Extracts a **9-dimensional feature vector** for each candidate resource:
     1. **BM25 Relevance** (Okapi BM25 exact term matching)
     2. **TF-IDF Semantic Similarity** (n-gram sublinear cosine similarity)
     3. **Difficulty Match** ($1 - |\text{concept\_diff} - \text{resource\_diff}| / 4$)
     4. **Freshness** (publication recency scoring)
     5. **Popularity** (log-scale view and engagement metrics)
     6. **Source Authority** (MIT OCW = 1.0, Wikipedia = 0.90, YouTube = 0.75)
     7. **Graph Relevance** (topological predecessor/successor alignment)
     8. **Duration Fit** (micro-learning sweet spot: 10–25 mins)
     9. **Channel Authority** (verified top educational creators: 3Blue1Brown, MIT, Stanford, freeCodeCamp, etc.)
   - Computes weighted **Ensemble Relevance Scoring**.
   - Applies **Maximal Marginal Relevance (MMR) Diversification** ($\lambda = 0.65$) to eliminate redundant items and curate the **Top 3 Relevant and Diverse Resources** (e.g. 1 video, 1 article, 1 course syllabus) bound directly to each concept node.
4. **Domain-Specific GNN Representation Learning**:
   - Constructs a **10-dimensional node feature matrix** $X \in \mathbb{R}^{|\mathcal{V}| \times 10}$ fusing graph topology, learner mastery state, and the ML resource signals (BM25, TF-IDF, Resource Authority).
   - Executes 2-layer **Graph Convolutional Networks (GCN)** or **Graph Attention Networks (GAT)** to project concepts into neighborhood-aware embedding space.
   - Computes composite recommendation scores combining **Prerequisite Readiness**, **Target Relevance** (embedding + TF-IDF/BM25), and **Marginal Information Gain (MIG)** with strict prerequisite-constrained filtering.
5. **Agentic Tutoring & Micro-Lesson Delivery (Phase C)**:
   - LangGraph orchestrator prompts Gemini to synthesize focused micro-lessons grounded directly in the concept's Top-3 MMR-diversified resources.
   - Generates 3 multiple-choice mastery check questions with distractor rationales.
6. **Mastery Updates & Real-Time Re-ranking**:
   - Deterministic grading updates the learner's dynamic mastery vector $q_u$ upon passing ($\ge 70\%$).
   - Cascades unlocks through the DAG and triggers instant GNN re-ranking for subsequent steps.

---

## 2. Full Technology Stack

### A. Backend & Core Architecture
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.11 / 3.14 | Core execution environment |
| **FastAPI** | 0.115.0 | Asynchronous REST API framework |
| **Uvicorn** | 0.30.6 | ASGI web server |
| **Pydantic v2** | 2.9.2 | Strict schema validation, data serialization, and API contract enforcement |
| **python-dotenv** | 1.0.1 | Configuration and environment variable management |

### B. AI & Agentic Orchestration
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Google GenAI SDK** | 2.8.0 | Native interface to Gemini 2.5 Flash for curriculum synthesis, lesson generation, and grading |
| **LangGraph** | 0.2.34 | Stateful workflow orchestration for lesson generation $\to$ quiz creation $\to$ evaluation cycles |
| **LangChain Core** | 0.3.5 | Prompt abstraction and structured output schemas |

### C. Knowledge Graph Engine
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **NetworkX** | 3.3 / 3.6 | Graph topology representation, BFS depth calculation, cycle detection, and topological sorting |
| **Custom DAG Validator** | Internal | Cycle-breaking fallback, isolated node pruning, and initial mastery state mapping |

### D. Machine Learning & Resource Ranking Engine
| Component | Implementation | Description |
| :--- | :--- | :--- |
| **BM25 Ranker** | Okapi BM25 (`app/ranking/bm25.py`) | Term frequency & inverse document frequency matching ($k_1=1.5, b=0.75$) |
| **TF-IDF Ranker** | `TFIDFSimilarityRanker` (`app/ranking/tfidf_ranker.py`) | Scikit-Learn n-gram (1-2) vectorizer with sublinear TF and cosine similarity |
| **9-Feature Extractor** | `app/ranking/resource_ranker.py` | BM25, TF-IDF, Difficulty Match, Freshness, Popularity, Authority, Graph, Duration, Channel |
| **Ensemble Scoring** | Weighted Linear Model | Multi-feature normalized relevance weighting |
| **Diversification** | Maximal Marginal Relevance (MMR) | Balances relevance with inter-resource diversity ($\lambda=0.65$) to avoid duplicate media |

### E. Graph Neural Network (GNN) Engine
| Component | Implementation | Description |
| :--- | :--- | :--- |
| **Primary GNN** | 2-Layer GCN (`GCNConv`) | Spectral graph convolutions: $H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$ |
| **Alternative GNN** | 2-Layer GAT (`GATConv`) | Multi-head graph attention network capturing anisotropic prerequisite relevance |
| **Spectral NumPy Fallback** | `NumpyGCN` | Pure NumPy closed-form spectral GCN with perturbation-based BPR training (zero binary dependency) |
| **Training Objective** | Pairwise BPR Ranking Loss | $\mathcal{L}_{\text{BPR}} = -\log \sigma(s(u, v^+) - s(u, v^-))$ optimized over synthetic learner trajectories |

### F. Scientific Evaluation & Analytics
| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Scikit-Learn** | 1.9.1 | Embedding analysis, cosine similarity metrics, and community clustering |
| **SciPy** | 1.17.1 | Sparse linear algebra and distance computation |
| **NumPy** | 1.26 / 2.5 | Matrix operations, vectorized scoring, and tensor transformations |
| **Plotly** | 7.1.0 | Interactive 2D t-SNE projections, metric radars, and multi-depth accuracy curves |

### G. Frontend & User Interfaces
| Layer | Tech Stack | Role |
| :--- | :--- | :--- |
| **Streamlit Interactive UI** | Streamlit 1.38, `streamlit-mermaid` | Comprehensive 3-tab portal: Interactive Journey & Tutor, GNN Recommendations, and Scientific Evaluation Suite |
| **Research Dashboard** | Streamlit, Plotly, Pandas | Journal-grade evaluation dashboard with loss curves, model win tables, and t-SNE embedding visualizer |
| **Modern Web Application** | React 18, TypeScript, Vite, Tailwind CSS, Framer Motion, Radix UI, XYFlow | Full production-grade SPA with interactive graph nodes and responsive lesson delivery |

### H. Ingestion Sources
| Source | Client / Library | Ingested Data |
| :--- | :--- | :--- |
| **Wikipedia** | Python Requests (REST API) | Domain summaries, concept taxonomy, and verified academic references |
| **YouTube Tutorials** | `google-api-python-client`, `youtube-transcript-api` | Video titles, channel metadata, and lesson transcripts |
| **MIT OpenCourseWare** | `feedparser` | University course syllabi, lecture readings, and problem sets |

---

## 3. Mathematical Formulation

### 3.1 Graph Representation & Node Features
The learning domain is modeled as a directed, attributed graph:
$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, X)$$
where $\mathcal{V}$ represents concept nodes, $\mathcal{E}$ represents directed prerequisite relationships, and $X \in \mathbb{R}^{|\mathcal{V}| \times 10}$ is the 10-dimensional node feature matrix:

1. **Difficulty ($X_{v,0}$)**: Normalized difficulty level: $\frac{\text{difficulty}}{5} \in [0.2, 1.0]$.
2. **Topological Depth ($X_{v,1}$)**: Normalized BFS level from root concepts: $\frac{\text{depth}(v)}{\max(\text{depth})}$.
3. **Topic Cluster ($X_{v,2}$)**: Community assignment computed via graph partitioning.
4. **Learner Mastery ($X_{v,3}$)**: Dynamic mastery score $m_{uv} \in [0.0, 1.0]$ updated after each quiz.
5. **In-Degree ($X_{v,4}$)**: Normalized count of incoming prerequisite dependencies.
6. **Out-Degree ($X_{v,5}$)**: Normalized count of dependent successor concepts.
7. **Prerequisite Ratio ($X_{v,6}$)**: Structural centrality ratio: $\frac{\text{in\_deg}}{\text{in\_deg} + \text{out\_deg} + 10^{-6}}$.
8. **BM25 Relevance ($X_{v,7}$)**: Mean normalized BM25 score of the node's curated resources against the curriculum query.
9. **TF-IDF Similarity ($X_{v,8}$)**: Mean cosine similarity of the node's curated resources against the learning goal.
10. **Resource Authority ($X_{v,9}$)**: Mean pedagogical authority & quality score of the top-3 curated resources.

### 3.2 Okapi BM25 Formulation
For concept query $Q$ and candidate resource document $D$:
$$\text{BM25}(D, Q) = \sum_{q_i \in Q} \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1.0\right) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
with $k_1 = 1.5$ and $b = 0.75$.

### 3.3 Maximal Marginal Relevance (MMR) Diversification
Given candidate pool $R$ and selected resource set $S$:
$$\text{MMR} = \arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot \text{Score}_{\text{ensemble}}(D_i, C) - (1 - \lambda) \cdot \max_{D_j \in S} \text{Sim}(D_i, D_j) \right]$$
with $\lambda = 0.65$. This penalizes textual and media-type redundancy, guaranteeing diverse multi-modal support (video + article + course notes).

### 3.4 Composite Recommendation Scoring Function
Candidate concepts are scored using a weighted multi-factor objective:
$$\text{Score}(u, v) = \alpha \cdot \text{Readiness}(u, v) + \beta \cdot \text{Relevance}(v, g) + \gamma \cdot \text{MIG}(u, v)$$
where:
* **Prerequisite Readiness ($\text{Readiness}(u, v)$)**: Mean mastery of $v$'s immediate prerequisite predecessors:
  $$\text{Readiness}(u, v) = \begin{cases} 1.0 & \text{if } \text{Pred}(v) = \emptyset \\ \frac{1}{|\text{Pred}(v)|} \sum_{p \in \text{Pred}(v)} m_{up} & \text{otherwise} \end{cases}$$
* **Effective Target Relevance ($\text{Relevance}(v, g)$)**: Fusion of GNN embedding direction with BM25 & TF-IDF semantic relevance:
  $$\text{Relevance}(v, g) = 0.60 \cdot \text{Sim}_{\text{cos}}(h_v, g_{\text{GNN}}) + 0.25 \cdot \text{TFIDF}(v) + 0.15 \cdot \text{BM25}(v)$$
* **Marginal Information Gain ($\text{MIG}(u, v)$)**: Potential knowledge advancement:
  $$\text{MIG}(u, v) = (1.0 - m_{uv}) \cdot \frac{\text{difficulty}_v}{5}$$
* **Default Coefficients**: $\alpha = 0.40$, $\beta = 0.35$, $\gamma = 0.25$.

### 3.5 Strict Prerequisite Constraint Filtering
To eliminate premature cognitive overload, PathBuilder applies strict prerequisite-aware filtering:
$$\text{Eligible}(u, v) = \mathbf{1}\left(\forall p \in \text{Pred}(v): m_{up} \ge \tau_{\text{pass}}\right) \quad \text{where } \tau_{\text{pass}} = 0.70$$
Concepts failing this constraint are marked as `locked` and appended after eligible candidates regardless of raw model score.

---

## 4. End-to-End System Workflow

```mermaid
flowchart TD
    subgraph Frontend [Frontend Layer: Streamlit / React]
        UI1[Goal Input]
        UI2[Interactive Knowledge Graph]
        UI3[Adaptive Tutor & Quiz View]
        UI4[Progress Tracking & Resource Links]
    end

    subgraph Backend [FastAPI Backend]
        API1[/build-path]
        API2[/graph]
        API3[/tutor/next]
        API4[/tutor/answer]
        API5[/gnn/recommend]
        STORE[(Session & Mastery Store)]
    end

    subgraph PhaseA [Phase A: Ingestion Agent]
        WIKI[Wikipedia API]
        YT[YouTube Data API]
        OCW[MIT OpenCourseWare RSS]
        NORM[Resource Normalization: Video | Article | Course]
    end

    subgraph PhaseB [Phase B: Graph Builder Agent]
        GEMINI_KG[Gemini Concept & DAG Generation]
        PYDANTIC[Pydantic Schema Validation]
        NX[NetworkX Cycle Detection & DAG Checks]
    end

    subgraph ML_Ranker [Resource Ranking Pipeline]
        FEAT[9 Feature Extraction: BM25, TF-IDF, Diff, Auth, Dur, Fresh, Pop...]
        ENSEMBLE[Ensemble Relevance Scoring]
        MMR[MMR Diversification λ=0.65]
        TOP3[Top-3 Diverse Resources per Concept]
    end

    subgraph GNN_Layer [GNN Recommendation Engine]
        FEAT_MAT[Node Feature Matrix X ∈ R^{N×10}]
        GCN[2-Layer GCN / GAT Forward Pass]
        SCORE[Composite Scoring: Readiness + Relevance + MIG]
        FILTER[Prerequisite Constraint Filtering]
    end

    subgraph PhaseC [Phase C: Content Generator & Adaptive Tutor]
        LESSON[Gemini Micro-Lesson Grounded in Top-3 Resources]
        QUIZ[Quiz Generation: 3 MCQs]
        EVAL[Deterministic Answer Grading]
        UNLOCK[Mastery Update & Graph Unlock Cascade]
    end

    UI1 --> API1 --> WIKI & YT & OCW --> NORM
    NORM --> GEMINI_KG --> PYDANTIC --> NX
    NX --> FEAT --> ENSEMBLE --> MMR --> TOP3
    TOP3 --> FEAT_MAT --> GCN --> SCORE --> FILTER
    FILTER --> LESSON --> QUIZ --> UI3
    UI3 --> API4 --> EVAL --> UNLOCK --> STORE
    UNLOCK --> FEAT_MAT
```

---

## 5. Scientific Evaluation & Benchmarking

PathBuilder includes a scientific evaluation harness comparing the proposed GNN architecture against three established baselines:

### 5.1 Baseline Implementations
1. **Topological Sort Baseline**: Static greedy traversal ordered strictly by topological order (in-degree roots first).
2. **Heuristic Ranker Baseline**: Ranks eligible concepts using a linear combination of normalized topological depth and difficulty.
3. **Random Walk with Restart (PageRank)**: Simulates personalized random walks on the prerequisite graph with a restart probability $p_{\text{restart}} = 0.15$.

### 5.2 Evaluation Metrics
* **NDCG@K (Normalized Discounted Cumulative Gain)**:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
* **MRR (Mean Reciprocal Rank)**:
  $$\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{i=1}^{|\mathcal{Q}|} \frac{1}{\text{rank}_i}$$
* **Prerequisite Violation Rate (PVR)**:
  $$\text{PVR} = \frac{1}{|P|} \sum_{v \in P} \mathbf{1}\left(\exists p \in \text{Pred}(v): p \notin \text{VisitedBefore}(v)\right)$$

### 5.3 Empirical Results (Automated Benchmark Suite)
| Model Architecture | NDCG@5 | NDCG@10 | MRR | PVR (Lower is better) |
| :--- | :---: | :---: | :---: | :---: |
| **GNN (Proposed System + ML Features)** | **0.9197** | **0.9197** | **1.0000** | **0.0000** (Prerequisite-Constrained) |
| **Topological Sort** | 0.5706 | 0.5706 | 0.3333 | 0.0000 |
| **Heuristic Ranker** | 0.8240 | 0.8240 | 0.7500 | 0.2500 |
| **Random Walk with Restart** | 0.2641 | 0.4825 | 0.2500 | 0.8750 |

---

## 6. How to Run the System

### 1. Launch FastAPI Backend
```bash
.\venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --port 8000
```
* API Documentation: `http://localhost:8000/docs`
* Health Endpoint: `http://localhost:8000/health`

### 2. Launch Streamlit Learning Portal
```bash
.\venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py --server.port 8501
```
* Access the interactive learner UI at `http://localhost:8501`.

### 3. Launch Research Evaluation Dashboard
```bash
.\venv\Scripts\python.exe -m streamlit run evaluation/dashboard.py --server.port 8502
```
* Access the journal evaluation dashboard at `http://localhost:8502`.

### 4. Generate Journal Paper Results & Figures (300 DPI)
```bash
.\venv\Scripts\python.exe generate_paper_results.py
```
* Runs the multi-seed evaluation suite across 3 benchmark curricula (*Advanced Data Structures & Algorithms*, *Modern Machine Learning & GNNs*, *Quantum Computing Fundamentals*).
* Automatically exports publication-ready 300 DPI figures and ready-to-paste LaTeX tables into `paper_artifacts/`:
  - `paper_artifacts/fig1_training_loss.png`: Pairwise BPR training loss convergence over 50 epochs.
  - `paper_artifacts/fig2_model_comparison.png`: Grouped comparative bar chart (NDCG@5, NDCG@10, MRR).
  - `paper_artifacts/fig3_depth_accuracy.png`: Accuracy curves across topological depths.
  - `paper_artifacts/fig4_ablation_study.png`: Component ablation study degradation chart.
  - `paper_artifacts/fig5_tsne_embeddings.png`: 2D t-SNE manifold visualization of learned concept representations.
  - `paper_artifacts/JOURNAL_RESULTS.md`: Complete formatted LaTeX tables (`\begin{table}...`) and Markdown summary.

### 5. Run Automated Test Suites
```bash
# End-to-end GNN architecture, features, BM25, TF-IDF, MMR, scoring, baselines, and evaluation
.\venv\Scripts\python.exe -m unittest test_gnn_pipeline.py -v

# FastAPI REST endpoint integration tests
.\venv\Scripts\python.exe -m unittest test_api_endpoints.py -v
```

---

## 7. API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/build-path` | Ingests sources, constructs DAG, ranks resources with BM25/TF-IDF/MMR, initializes session |
| `GET` | `/graph/{session_id}` | Retrieves graph nodes with Top-3 diverse resources, edges, Mermaid diagram, and progress |
| `GET` | `/tutor/next/{session_id}` | Fetches personalized micro-lesson grounded in Top-3 resources + quiz for current concept |
| `POST` | `/tutor/answer` | Grades submitted quiz answers and updates mastery state |
| `GET` | `/gnn/recommend/{session_id}` | Returns real-time GNN-ranked concept recommendations with ML scores and curated resources |
| `POST` | `/gnn/train/{session_id}` | Trains GCN on synthetic trajectories using pairwise BPR loss |
| `GET` | `/gnn/evaluate/{session_id}` | Computes NDCG@K, MRR, PVR, and ablation metrics across all models |
| `GET` | `/gnn/baselines/{session_id}` | Returns Topological, Heuristic, and Random Walk rankings side-by-side |
| `GET` | `/health` | Service liveness probe |
