"""
Streamlit UI for the Agentic AI Learning Path Builder with GNN Recommendation Engine.

Tabs:
  1. 🧭 Learning Journey & Tutor: Prerequisite graph, micro-lessons, interactive quizzes.
  2. 🧠 GNN Recommendation Engine: Dynamic ranking, readiness scores, baseline comparisons, training.
  3. 📊 Scientific Evaluation Portal: NDCG@K, MRR, PVR, ablations, and per-depth analysis.
"""
import os
import requests
import streamlit as st
import pandas as pd
from streamlit_mermaid import st_mermaid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PathBuilder – GNN Learning Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling for Premium Look & Feel
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-box {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
  }
  .metric-title {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .rec-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "current_lesson" not in st.session_state:
    st.session_state.current_lesson = None
if "gnn_recs" not in st.session_state:
    st.session_state.gnn_recs = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

st.title("🧭 PathBuilder: Agentic GNN Learning System")
st.caption(
    "Prerequisite-aware, personalized learning journeys powered by Graph Neural Networks (GCN/GAT) "
    "and agentic multi-source ingestion (Wikipedia, YouTube, MIT OCW)."
)

# --------------------------------------------------------------------------
# Step 1: Goal Input
# --------------------------------------------------------------------------
with st.form("goal_form"):
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        goal = st.text_input(
            "What do you want to master?",
            placeholder="e.g. Quantum Computing, Graph Neural Networks, Distributed Systems",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Build Learning Path", use_container_width=True, type="primary")

if submitted and goal.strip():
    with st.spinner("Ingesting sources and constructing prerequisite DAG..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/build-path", json={"goal": goal}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.graph_data = data
            st.session_state.current_lesson = None
            st.session_state.gnn_recs = None
            st.session_state.eval_results = None
            st.success("Knowledge Graph synthesized successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to build learning path: {e}")

# --------------------------------------------------------------------------
# Step 2: Main Dashboard
# --------------------------------------------------------------------------
if st.session_state.session_id:
    session_id = st.session_state.session_id

    try:
        graph_resp = requests.get(f"{BACKEND_URL}/graph/{session_id}", timeout=30)
        graph_resp.raise_for_status()
        graph = graph_resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load graph: {e}")
        st.stop()

    tab_journey, tab_gnn, tab_eval = st.tabs([
        "🧭 Interactive Journey & Tutor",
        "🧠 GNN Recommendation Engine",
        "📊 Scientific Evaluation Portal"
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Interactive Journey & Tutor
    # ══════════════════════════════════════════════════════════════════════
    with tab_journey:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader(f"Knowledge Graph: {graph['goal']}")
            st_mermaid(graph["mermaid"], height=480)

            st.markdown("#### Learner Mastery Progression")
            progress = graph["progress"]
            cols = st.columns(3)
            mastered_count = sum(1 for s in progress["node_status"].values() if s == "mastered")
            avail_count = sum(1 for s in progress["node_status"].values() if s in ("available", "in_progress"))
            locked_count = sum(1 for s in progress["node_status"].values() if s == "locked")

            with cols[0]:
                st.markdown(f"<div class='metric-box'><div class='metric-value'>{mastered_count}</div><div class='metric-title'>Mastered</div></div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div class='metric-box'><div class='metric-value'>{avail_count}</div><div class='metric-title'>Ready / Active</div></div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div class='metric-box'><div class='metric-value'>{locked_count}</div><div class='metric-title'>Locked</div></div>", unsafe_allow_html=True)

            st.write("")
            for node in graph["nodes"]:
                status = progress["node_status"].get(node["id"], "locked")
                score = progress["node_scores"].get(node["id"])
                icon = {"locked": "🔒", "available": "🔓", "in_progress": "🟡", "mastered": "✅"}.get(status, "⬜")
                score_txt = f" — Score: {score*100:.0f}%" if score is not None else ""
                st.markdown(f"**{icon} {node['title']}** (Diff: {node['difficulty']}/5){score_txt}")

        with right:
            st.subheader("Adaptive Gemini Tutor")

            if st.button("Load Next Recommended Lesson", type="primary", use_container_width=True):
                with st.spinner("Generating focused micro-lesson and mastery quiz..."):
                    try:
                        r = requests.get(f"{BACKEND_URL}/tutor/next/{session_id}", timeout=90)
                        r.raise_for_status()
                        st.session_state.current_lesson = r.json()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to load lesson: {e}")

            lesson = st.session_state.current_lesson
            if lesson and lesson.get("done"):
                st.balloons()
                st.success(lesson.get("message", "All concepts mastered. Goal achieved!"))
            elif lesson:
                st.markdown(f"### {lesson['node_title']}  \n*Difficulty Level: {lesson['difficulty']}/5*")

                if lesson.get("resources"):
                    st.markdown("**Curated Multi-Source Materials (BM25 & TF-IDF):**")
                    icons = {"video": "🎥 Video", "article": "📖 Reading", "course": "🎓 Course", "pdf": "📄 Document"}
                    r_cols = st.columns(min(3, len(lesson["resources"])))
                    for idx, res in enumerate(lesson["resources"][:3]):
                        with r_cols[idx]:
                            tag = icons.get(res.get("resource_type", "article"), "📚 Resource")
                            st.markdown(f"[{tag}: {res.get('title')[:32]}...]({res.get('url')})")
                elif lesson.get("source_url"):
                    st.markdown(f"[📚 Source Reference Material]({lesson['source_url']})")

                st.info(lesson["lesson"])

                st.markdown("---")
                st.markdown("#### Concept Mastery Quiz")
                answers = []
                with st.form("quiz_form"):
                    for i, q in enumerate(lesson["quiz"]):
                        choice = st.radio(
                            f"**Q{i+1}: {q['question']}**",
                            options=list(range(len(q["options"]))),
                            format_func=lambda idx, opts=q["options"]: opts[idx],
                            key=f"quiz_{lesson['node_id']}_{i}",
                        )
                        answers.append(choice)
                    quiz_submitted = st.form_submit_button("Submit Mastery Check", type="primary")

                if quiz_submitted:
                    with st.spinner("Grading response and updating graph mastery..."):
                        try:
                            r = requests.post(
                                f"{BACKEND_URL}/tutor/answer",
                                json={
                                    "session_id": session_id,
                                    "node_id": lesson["node_id"],
                                    "user_answers": answers,
                                },
                                timeout=60,
                            )
                            r.raise_for_status()
                            result = r.json()
                            if result["passed"]:
                                st.success(f"Score: {result['score']*100:.0f}% — Concept Mastered! 🎉")
                            else:
                                st.warning(f"Score: {result['score']*100:.0f}% — Below mastery threshold (70%). Review and retry.")
                            for fb in result["feedback"]:
                                st.write(f"- {fb}")
                            if result["goal_complete"]:
                                st.balloons()
                                st.success("Curriculum Completed!")
                            st.session_state.current_lesson = None
                            st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Failed to submit answers: {e}")
            else:
                st.info("Click **Load Next Recommended Lesson** to begin tutoring on your first unlocked concept.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: GNN Recommendation Engine
    # ══════════════════════════════════════════════════════════════════════
    with tab_gnn:
        st.subheader("GNN-Ranked Concept Recommendations")
        st.caption(
            "Composite scoring formula:  "
            r"$Score(u, v) = \alpha \cdot \text{Readiness}(u, v) + \beta \cdot \text{Relevance}(v) + \gamma \cdot \text{MIG}(u, v)$"
        )

        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            refresh_recs = st.button("Refresh GNN Inference", use_container_width=True)
        with col_act2:
            train_btn = st.button("Train GNN on Session Graph (BPR Loss)", use_container_width=True)

        if train_btn:
            with st.spinner("Training GCN recommender with pairwise BPR ranking loss..."):
                try:
                    t_resp = requests.post(
                        f"{BACKEND_URL}/gnn/train/{session_id}",
                        json={"epochs": 40, "lr": 0.005, "n_learners": 60},
                        timeout=60,
                    )
                    t_resp.raise_for_status()
                    t_data = t_resp.json()
                    st.success(f"GNN trained in {t_data['training']['training_time_s']}s (Final Loss: {t_data['training']['final_train_loss']})")
                    refresh_recs = True
                except Exception as ex:
                    st.error(f"Training failed: {ex}")

        # Fetch recommendations
        try:
            rec_resp = requests.get(
                f"{BACKEND_URL}/gnn/recommend/{session_id}?force_refresh={refresh_recs}",
                timeout=30
            )
            rec_resp.raise_for_status()
            recs_data = rec_resp.json()
            st.session_state.gnn_recs = recs_data
        except Exception as ex:
            st.error(f"Failed to fetch GNN recommendations: {ex}")

        if st.session_state.gnn_recs:
            recs = st.session_state.gnn_recs
            top_rec = recs.get("recommended_next")
            top_concept = next((r for r in recs["ranked_concepts"] if r["node_id"] == top_rec), None)

            if top_concept:
                res_html = ""
                if top_concept.get("resources"):
                    res_html = "<div style='margin-top:10px; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;'><b style='color:#a5b4fc;'>Curated Resources (MMR Diversified):</b><br/>"
                    icons = {"video": "🎥", "article": "📖", "course": "🎓", "pdf": "📄"}
                    for r in top_concept["resources"]:
                        ic = icons.get(r.get("resource_type", "article"), "📌")
                        res_html += f"<div style='margin-top:4px; font-size:0.85rem;'>{ic} <a href='{r.get('url')}' target='_blank' style='color:#38bdf8; text-decoration:none;'>{r.get('title')}</a> <span style='color:#94a3b8;'>({r.get('resource_type').upper()} • BM25: {r.get('bm25_score', 0):.2f} • TF-IDF: {r.get('tfidf_score', 0):.2f})</span></div>"
                    res_html += "</div>"

                st.markdown(f"""
                <div class="rec-card">
                  <span style="background:#6366f1; color:white; font-size:0.75rem; font-weight:700; padding:3px 8px; border-radius:6px;">
                    ⭐ TOP GNN RECOMMENDATION
                  </span>
                  <h3 style="margin-top:8px; margin-bottom:4px;">{top_concept['title']} (Difficulty: {top_concept['difficulty']}/5)</h3>
                  <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:6px;">
                    Composite Score: <b>{top_concept['score']}</b> | Prerequisite Readiness: <b>{top_concept['prerequisite_readiness']*100:.0f}%</b> | Target Relevance: <b>{top_concept['target_relevance']*100:.0f}%</b> | Marginal Gain: <b>{top_concept['marginal_gain']}</b>
                  </p>
                  <p style="color:#cbd5e1; font-size:0.85rem; margin-bottom:8px;">
                    ML Signals: BM25 Relevance: <b>{top_concept.get('bm25_score', 0.0)}</b> | TF-IDF Similarity: <b>{top_concept.get('tfidf_score', 0.0)}</b> | Source Authority: <b>{top_concept.get('resource_quality', 0.0)}</b>
                  </p>
                  {res_html}
                </div>
                """, unsafe_allow_html=True)

            # Table of all concepts
            df_recs = pd.DataFrame(recs["ranked_concepts"])
            display_cols = ["title", "difficulty", "score", "prerequisite_readiness", "target_relevance", "bm25_score", "tfidf_score", "marginal_gain", "status", "eligible"]
            valid_cols = [c for c in display_cols if c in df_recs.columns]
            if valid_cols:
                rename_map = {
                    "title": "Concept",
                    "difficulty": "Diff (1-5)",
                    "score": "GNN Score",
                    "prerequisite_readiness": "Readiness",
                    "target_relevance": "Goal Relevance",
                    "bm25_score": "BM25 Rel",
                    "tfidf_score": "TF-IDF Sim",
                    "marginal_gain": "MIG",
                    "status": "State",
                    "eligible": "Eligible"
                }
                st.dataframe(
                    df_recs[valid_cols].rename(columns={k: rename_map[k] for k in valid_cols if k in rename_map}),
                    use_container_width=True,
                )

        # Baseline Comparison Sub-Section
        with st.expander("Compare with Conventional Baselines", expanded=False):
            try:
                base_resp = requests.get(f"{BACKEND_URL}/gnn/baselines/{session_id}", timeout=20)
                if base_resp.ok:
                    base_data = base_resp.json()
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        st.markdown("**Topological Sort**")
                        for i, item in enumerate(base_data.get("topological", [])[:5]):
                            st.write(f"{i+1}. {item['title']}")
                    with b_col2:
                        st.markdown("**Heuristic Ranker**")
                        for i, item in enumerate(base_data.get("heuristic", [])[:5]):
                            st.write(f"{i+1}. {item['title']}")
                    with b_col3:
                        st.markdown("**Random Walk / PageRank**")
                        for i, item in enumerate(base_data.get("random_walk", [])[:5]):
                            st.write(f"{i+1}. {item['title']}")
            except Exception as e:
                st.caption(f"Baselines preview unavailable: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Scientific Evaluation Portal
    # ══════════════════════════════════════════════════════════════════════
    with tab_eval:
        st.subheader("Journal-Grade Benchmark & Scientific Evaluation")
        st.caption(
            "Empirical evaluation comparing GNN Recommender against 3 baselines and ablation variants."
        )

        run_eval = st.button("Run Evaluation Benchmark Suite", type="primary")

        if run_eval or st.session_state.eval_results is None:
            with st.spinner("Computing NDCG@5, NDCG@10, MRR, PVR, and Ablation Study..."):
                try:
                    eval_resp = requests.get(f"{BACKEND_URL}/gnn/evaluate/{session_id}?seed=42", timeout=60)
                    eval_resp.raise_for_status()
                    st.session_state.eval_results = eval_resp.json()
                except Exception as ex:
                    st.error(f"Evaluation benchmark failed: {ex}")

        if st.session_state.eval_results:
            results = st.session_state.eval_results
            models = results.get("models", {})

            # Metric Table
            st.markdown("#### 1. Model Performance Comparison")
            rows = []
            for name, m in models.items():
                rows.append({
                    "Model Architecture": name,
                    "NDCG@5": f"{m.get('ndcg@5', 0.0):.4f}",
                    "NDCG@10": f"{m.get('ndcg@10', 0.0):.4f}",
                    "MRR (Mean Recip Rank)": f"{m.get('mrr', 0.0):.4f}",
                    "Prereq Violation Rate (PVR)": f"{m.get('pvr', 0.0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Ablation Analysis
            st.markdown("#### 2. Ablation Study")
            ablations = results.get("ablations", {})
            abl_rows = []
            for name, m in ablations.items():
                abl_rows.append({
                    "Ablation Variant": name,
                    "NDCG@5": f"{m.get('ndcg@5', 0.0):.4f}",
                    "MRR": f"{m.get('mrr', 0.0):.4f}",
                    "PVR": f"{m.get('pvr', 0.0):.4f}",
                })
            st.dataframe(pd.DataFrame(abl_rows), use_container_width=True)

            # Per-Depth Accuracy
            st.markdown("#### 3. Recommendation Accuracy by Prerequisite Depth")
            depth_data = results.get("per_depth", {})
            if depth_data:
                depth_rows = []
                for d, scores in depth_data.items():
                    d_row = {"Prerequisite Depth": f"Level {d}"}
                    d_row.update({k: f"{v:.4f}" for k, v in scores.items()})
                    depth_rows.append(d_row)
                st.dataframe(pd.DataFrame(depth_rows), use_container_width=True)

            st.info(
                "💡 **Key Finding**: The prerequisite-constrained GNN maintains superior ranking quality "
                "(high NDCG & MRR) while achieving near-zero Prerequisite Violation Rates (PVR) compared "
                "to heuristic or random walk baselines."
            )
else:
    st.info("Enter a learning goal above to synthesize your prerequisite graph and start learning!")
