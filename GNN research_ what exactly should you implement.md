## **GNN research: what exactly should you implement?**

This is the most important technical decision. Don't build a GNN just to say the project uses one. It should improve a measurable part of PathBuilder.

I recommend framing the research question as:

> Can graph neural networks improve prerequisite-aware, personalized learning-path recommendation compared with conventional resource ranking and graph-based baselines?

Your model should predict or rank the next concept to learn, considering both graph structure and the learner's current state.

### **A. Define the graph mathematically**

Represent the learning domain as a directed, attributed graph:

$G=(V,E,X)$

where:

* $V$: skills, concepts and optionally learning resources.  
* $E$: prerequisite, similarity and resource-to-concept relationships.  
* $X$: node features such as semantic embeddings, difficulty, estimated duration and resource quality.

For example:

Arrays

Mastery: 0.90

Hashing

Mastery: 0.45

Sliding Window

Mastery: 0.10

Illustrative graph only. Actual prerequisite edges must be validated for the selected curriculum.

Each learner also has a state vector:

${q}_{u}=[{m}_{u1},{m}_{u2},\ldots ,{m}_{u|V|}]$

Here, ${m}_{uv}$ represents the estimated mastery of concept $v$ for learner $u$. You can extend this state with attempts, recent quiz performance and time spent.

### **B. Implement a baseline GNN first**

A two-layer Graph Convolutional Network (GCN) is a good starting point:

${H}^{(l+1)}=\sigma \left({{\tilde{D}}^{-1/2}\tilde{A}{\tilde{D}}^{-1/2}{H}^{(l)}{W}^{(l)}}\right)$

where:

* $\tilde{A}=A+I$ adds self-connections.  
* $\tilde{D}$ is the degree matrix of $\tilde{A}$.  
* ${H}^{(0)}=X$ contains the initial node features.  
* ${W}^{(l)}$ contains trainable parameters.  
* $\sigma$ is a nonlinear activation function.

The resulting embedding ${h}_{v}$ represents a concept in the context of its graph neighborhood, not just its own features.

### **C. Turn embeddings into a recommendation**

One possible scoring function is:

$s(u,v)={q}_{u}^{\top }{W}_{r}{h}_{v}+{b}_{v}$

Then use the scores to rank eligible concepts, with prerequisite constraints filtering out concepts whose required predecessors are not sufficiently mastered.

Train with a ranking or classification objective. For example, a pairwise ranking loss can encourage the actual next concept ${v}^{+}$ to score above an unchosen negative candidate ${v}^{-}$:

\\\[ \\mathcal L=-\\log\\sigma\\left( s(u,v^+)-s(u,v^-)\\right) \\\]

This is a proposed formulation for your project, not a claim that it reproduces any one published paper.

The contribution: compare this model against simpler methods and determine whether its graph-aware representations actually improve the journeys.

### **D. Evaluate it scientifically**

| Model | Purpose |
| ----- | ----- |
| Popularity or heuristic ranking | Simple baseline |
| TF-IDF/content-based ranking | Semantic relevance baseline |
| GCN recommender | Main GNN baseline |
| GraphSAGE or GAT | Alternative graph-learning architecture |
| GNN \+ prerequisite-constrained planner | Full proposed system |

Use metrics such as:

* Recommendation: Recall@K, Precision@K, NDCG@K.  
* Path quality: prerequisite violation rate, goal coverage and path length.  
* Learning outcome: simulated or observed mastery gain, with clear limitations.  
* Engineering: latency, resource coverage and graph-construction success rate.

Avoid claiming that the system improves actual learning outcomes based only on synthetic learners. If you lack real learner data, report offline recommendation metrics and carefully labelled simulations.

## **Journal paper: what should the team aim to publish?**

## **Title: PathBuilder: Graph Neural Network-Based Personalized Learning Journey Generation with Prerequisite-Aware Adaptation**

Potential research contributions

1. A graph representation that combines concept prerequisites, semantic similarity and resource-to-concept links.  
2. A GNN-based method for ranking the next concepts using graph structure and learner mastery.  
3. A prerequisite-aware planner integrated with an agentic tutor.  
4. An experimental comparison of conventional ranking, GNN variants and the full system.

To establish novelty, read these three papers closely before finalizing the method:

* [KnowLP — GraphRAG-Induced Dual Knowledge Structure Graphs for Personalized Learning Path Recommendation](https://doi.org/10.1609/aaai.v40i17.38479)  
* [LIGHT — Knowledge Topology-Aware Sequence Optimization](https://doi.org/10.1145/3726302.3730022)  
* [Towards Scalable Adaptive Learning with Graph Neural Networks and Reinforcement Learning](https://doi.org/10.48550/arXiv.2305.06398)

Your paper must clearly explain what is different from these existing methods. Integrating existing techniques into an application is a valid engineering achievement, but a journal research contribution needs a defensible methodological or empirical advance.

## **Six-week execution roadmap \- Individual Contributions Equally distributed** 

This schedule assumes approximately six weeks of work; it can be compressed or expanded around your actual capstone deadline.

## **W1**

Foundation

Scope, literature and interfaces

* All three: finalize the research question, related work and architecture.  
* Navneet: define graph schema and sample dataset.  
* Srikanth: choose the baseline GNN and formulate the recommendation objective.  
* Sanjay: define API contracts and the end-to-end user workflow.

Deliverable: approved problem statement, paper shortlist, architecture diagram and interface specification.

## **W2**

Data

Build the knowledge graph

* Navneet: ingestion, normalization, edge generation and graph validation.  
* Srikanth: node/edge feature design, preprocessing and train/test split strategy.  
* Sanjay: tutor and API skeleton, mock recommendation integration.

Deliverable: reproducible graph dataset with feature definitions and quality checks.

## **W3**

Model

Build baselines and GNN

* Srikanth: implement heuristic baseline, GCN and recommendation scoring.  
* Navneet: validate graph splits, labels and prerequisite correctness.  
* Sanjay: implement quiz scoring, learner-state persistence and mastery updates.

Deliverable: first runnable recommendation model and adaptive-tutor workflow.

## **W4**

Integration

Connect the complete journey

* Srikanth: expose recommendation scores through a stable interface.  
* Navneet: connect ranked resources to graph concepts.  
* Sanjay: integrate frontend, FastAPI, Gemini and learner progress.

Deliverable: goal input → graph → recommended path → lesson → quiz → updated mastery.

## **W5**

Evaluation

Experiments and paper draft

* Srikanth: model comparisons, ablations and statistical analysis.  
* Navneet: graph-quality results and dataset documentation.  
* Sanjay: end-to-end tests, latency and demonstration scenarios.

Deliverable: results tables, reproducible experiment scripts and a complete first paper draft.

## **W6**

Submission

PPT, reports and final validation

* All three: review paper claims and rehearse the demo.  
* Each member: prepare an individual contribution report backed by evidence.  
* Share responsibility for slides, limitations, references and future work.

Deliverable: final capstone PPT, individual reports, paper submission-ready draft and demo.

