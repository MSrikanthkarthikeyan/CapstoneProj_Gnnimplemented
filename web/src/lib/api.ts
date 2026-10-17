export const BACKEND_URL = "/api";

export interface ConceptNode {
  id: string;
  title: string;
  description: string;
  difficulty: number;
  source_url: string | null;
}

export interface PrerequisiteEdge {
  source: string;
  target: string;
  relation: string;
}

export interface KnowledgeGraph {
  goal: string;
  nodes: ConceptNode[];
  edges: PrerequisiteEdge[];
  mermaid: string;
  progress: {
    node_status: Record<string, "locked" | "available" | "in_progress" | "mastered">;
    node_scores: Record<string, number>;
    current_node: string | null;
  };
}

export interface LessonQuiz {
  done: boolean;
  message?: string;
  node_id?: string;
  node_title?: string;
  node_description?: string;
  difficulty?: number;
  source_url?: string;
  lesson?: string;
  quiz?: Array<{
    question: string;
    options: string[];
  }>;
}

export interface AnswerResult {
  score: number;
  passed: boolean;
  feedback: string[];
  next_node_id: string | null;
  goal_complete: boolean;
  mastery_snapshot: {
    node_status: Record<string, string>;
    node_scores: Record<string, number>;
  };
}

export const api = {
  buildPath: async (goal: string): Promise<{ session_id: string } & KnowledgeGraph> => {
    const res = await fetch(`${BACKEND_URL}/build-path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, max_youtube_results: 3, max_ocw_results: 5 }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to build path");
    }
    return res.json();
  },

  getGraph: async (session_id: string): Promise<KnowledgeGraph> => {
    const res = await fetch(`${BACKEND_URL}/graph/${session_id}`);
    if (!res.ok) throw new Error("Failed to load graph");
    return res.json();
  },

  getNextLesson: async (session_id: string): Promise<LessonQuiz> => {
    const res = await fetch(`${BACKEND_URL}/tutor/next/${session_id}`);
    if (!res.ok) throw new Error("Failed to load lesson");
    return res.json();
  },

  submitAnswer: async (
    session_id: string,
    node_id: string,
    user_answers: number[]
  ): Promise<AnswerResult> => {
    const res = await fetch(`${BACKEND_URL}/tutor/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, node_id, user_answers }),
    });
    if (!res.ok) throw new Error("Failed to submit answers");
    return res.json();
  },

  getGNNRecommend: async (session_id: string, force_refresh: boolean = false): Promise<any> => {
    const res = await fetch(`${BACKEND_URL}/gnn/recommend/${session_id}?force_refresh=${force_refresh}`);
    if (!res.ok) throw new Error("Failed to load GNN recommendations");
    return res.json();
  },

  trainGNN: async (
    session_id: string,
    params: { epochs?: number; lr?: number; n_learners?: number } = {}
  ): Promise<any> => {
    const res = await fetch(`${BACKEND_URL}/gnn/train/${session_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error("Failed to train GNN");
    return res.json();
  },

  getGNNEvaluate: async (session_id: string, seed: number = 42): Promise<any> => {
    const res = await fetch(`${BACKEND_URL}/gnn/evaluate/${session_id}?seed=${seed}`);
    if (!res.ok) throw new Error("Failed to evaluate models");
    return res.json();
  },

  getGNNBaselines: async (session_id: string): Promise<any> => {
    const res = await fetch(`${BACKEND_URL}/gnn/baselines/${session_id}`);
    if (!res.ok) throw new Error("Failed to load baselines");
    return res.json();
  },
};

