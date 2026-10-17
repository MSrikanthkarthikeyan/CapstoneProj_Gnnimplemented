import { create } from "zustand";
import { KnowledgeGraph, LessonQuiz } from "./api";

interface SessionState {
  sessionId: string | null;
  goal: string;
  graphData: KnowledgeGraph | null;
  currentLesson: LessonQuiz | null;
  isGenerating: boolean;
  setSession: (id: string, goal: string, graph: KnowledgeGraph) => void;
  updateGraph: (graph: KnowledgeGraph) => void;
  setCurrentLesson: (lesson: LessonQuiz | null) => void;
  setIsGenerating: (status: boolean) => void;
  reset: () => void;
}

export const useAppState = create<SessionState>((set) => ({
  sessionId: null,
  goal: "",
  graphData: null,
  currentLesson: null,
  isGenerating: false,
  setSession: (sessionId, goal, graphData) =>
    set({ sessionId, goal, graphData, currentLesson: null, isGenerating: false }),
  updateGraph: (graphData) => set({ graphData }),
  setCurrentLesson: (currentLesson) => set({ currentLesson }),
  setIsGenerating: (isGenerating) => set({ isGenerating }),
  reset: () =>
    set({ sessionId: null, goal: "", graphData: null, currentLesson: null, isGenerating: false }),
}));
