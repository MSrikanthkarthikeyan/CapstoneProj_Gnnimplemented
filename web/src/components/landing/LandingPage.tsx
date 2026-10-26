import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import { useAppState } from "@/lib/store";

const EXAMPLES = [
  "Master Quantum Computing from scratch",
  "Learn Rust Systems Programming",
  "Understand Transformer architectures",
  "Build distributed systems with Kafka",
  "Deep dive into Linear Algebra for ML",
];

function FloatingOrb({ x, y, size, color, delay }: { x: string; y: string; size: number; color: string; delay: number }) {
  return (
    <motion.div
      className="absolute rounded-full blur-3xl pointer-events-none"
      style={{ left: x, top: y, width: size, height: size, background: color, opacity: 0.12 }}
      animate={{ y: [0, -30, 0], scale: [1, 1.1, 1] }}
      transition={{ duration: 8 + delay, repeat: Infinity, ease: "easeInOut", delay }}
    />
  );
}

export function LandingPage() {
  const { setSession } = useAppState();
  const [goalInput, setGoalInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const goal = goalInput.trim();
    if (!goal) return;
    setError(null);
    setIsGenerating(true);
    setStatusMsg("Analyzing your learning goal...");

    const statusSteps = [
      [800,  "Fetching Wikipedia, YouTube & MIT OCW..."],
      [2200, "Building prerequisite knowledge graph with Gemini AI..."],
      [4000, "Running GNN inference & ranking concepts..."],
      [6000, "Personalizing your learning path..."],
    ] as const;

    const timers: ReturnType<typeof setTimeout>[] = [];
    statusSteps.forEach(([ms, msg]) => {
      timers.push(setTimeout(() => setStatusMsg(msg), ms));
    });

    try {
      const data = await api.buildPath(goal);
      timers.forEach(clearTimeout);
      setStatusMsg("Ready!");
      setSession(data.session_id, goal, data);
    } catch (err) {
      timers.forEach(clearTimeout);
      setError((err as Error).message || "Failed to build path. Please try again.");
      setIsGenerating(false);
      setStatusMsg("");
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden flex flex-col items-center justify-center px-4"
      style={{ background: "linear-gradient(135deg, hsl(220,25%,5%) 0%, hsl(222,25%,7%) 50%, hsl(220,22%,6%) 100%)" }}>

      {/* Background orbs */}
      <FloatingOrb x="10%" y="15%" size={500} color="hsl(38,95%,55%)" delay={0} />
      <FloatingOrb x="65%" y="60%" size={400} color="hsl(174,70%,42%)" delay={2} />
      <FloatingOrb x="40%" y="80%" size={300} color="hsl(38,70%,45%)" delay={4} />

      {/* Grid overlay */}
      <div className="absolute inset-0 pointer-events-none"
        style={{ backgroundImage: "radial-gradient(hsl(220,18%,25%,0.35) 1px, transparent 1px)", backgroundSize: "32px 32px" }} />

      {/* Vignette */}
      <div className="absolute inset-0 pointer-events-none"
        style={{ background: "radial-gradient(ellipse 80% 80% at 50% 50%, transparent 40%, hsl(220,25%,5%) 100%)" }} />

      {/* Main card */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-2xl"
      >
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.15 }}
          className="flex justify-center mb-8"
        >
          <span className="tag text-[10px] px-4 py-1.5 rounded-full"
            style={{ background: "hsl(38,95%,55%,0.12)", color: "hsl(38,90%,70%)", border: "1px solid hsl(38,95%,55%,0.25)" }}>
            ✦ Agentic AI • Graph Neural Networks • Adaptive Learning
          </span>
        </motion.div>

        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-center mb-10"
        >
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight mb-5 leading-[1.05]"
            style={{ letterSpacing: "-0.03em" }}>
            <span className="text-white">Learn anything,</span>
            <br />
            <span style={{
              background: "linear-gradient(135deg, hsl(38,95%,65%) 0%, hsl(38,90%,75%) 50%, hsl(174,70%,60%) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}>
              the right way.
            </span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-lg mx-auto leading-relaxed">
            Drop your learning goal — we build a personalized prerequisite knowledge graph, AI-generated micro-lessons, and adaptive mastery checks.
          </p>
        </motion.div>

        {/* Form */}
        <motion.form
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          onSubmit={handleSubmit}
          className="space-y-4"
        >
          <div className="relative">
            <input
              ref={inputRef}
              id="goal-input"
              value={goalInput}
              onChange={(e) => setGoalInput(e.target.value)}
              placeholder="What do you want to master?"
              disabled={isGenerating}
              className="w-full px-5 py-4 text-base rounded-xl text-white placeholder-[hsl(215,15%,40%)] outline-none transition-all duration-200 disabled:opacity-50"
              style={{
                background: "hsl(222,23%,11%)",
                border: goalInput ? "1.5px solid hsl(38,95%,55%,0.5)" : "1.5px solid hsl(220,18%,20%)",
                boxShadow: goalInput ? "0 0 0 4px hsl(38,95%,55%,0.06)" : "none",
              }}
              autoFocus
            />
            {goalInput && !isGenerating && (
              <button
                type="button"
                onClick={() => setGoalInput("")}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-white transition-colors text-lg"
              >×</button>
            )}
          </div>

          {/* Example chips */}
          {!goalInput && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => { setGoalInput(ex); inputRef.current?.focus(); }}
                  className="text-xs px-3 py-1.5 rounded-full transition-all duration-150 hover:scale-[1.02]"
                  style={{ background: "hsl(222,23%,13%)", border: "1px solid hsl(220,18%,20%)", color: "hsl(215,15%,60%)" }}
                >
                  {ex}
                </button>
              ))}
            </motion.div>
          )}

          <button
            type="submit"
            disabled={isGenerating || !goalInput.trim()}
            className="w-full py-4 rounded-xl text-base font-semibold transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed hover:scale-[1.01] active:scale-[0.99] cursor-pointer"
            style={{
              background: "linear-gradient(135deg, hsl(38,95%,52%) 0%, hsl(38,90%,60%) 100%)",
              color: "hsl(220,25%,7%)",
              boxShadow: isGenerating ? "none" : "0 4px 24px hsl(38,95%,55%,0.3), 0 1px 3px hsl(0,0%,0%,0.3)",
            }}
          >
            {isGenerating ? (
              <span className="flex items-center justify-center gap-3">
                <span className="w-4 h-4 rounded-full border-2 border-[hsl(220,25%,20%)] border-t-[hsl(220,25%,7%)] animate-spin" />
                {statusMsg}
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                Build My Learning Journey
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
            )}
          </button>

          <div className="flex items-center justify-center pt-1">
            <button
              type="button"
              onClick={() => {
                import("@/lib/demoData").then(({ DEMO_GRAPH }) => {
                  setSession("demo_gnn_session", DEMO_GRAPH.goal, DEMO_GRAPH);
                });
              }}
              className="text-xs text-[hsl(38,90%,70%)] hover:text-white transition-colors flex items-center gap-1.5 py-1.5 px-3 rounded-lg hover:bg-[hsl(220,18%,15%)] cursor-pointer"
            >
              <span>⚡ Or Explore Sample Knowledge Graph (Instant Demo)</span>
            </button>
          </div>

          {error && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="text-sm text-red-400 text-center py-2 px-4 rounded-lg"
              style={{ background: "hsl(0,60%,15%)", border: "1px solid hsl(0,60%,30%)" }}>
              {error}
            </motion.div>
          )}
        </motion.form>

        {/* Footer stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-10 grid grid-cols-3 gap-4"
        >
          {[
            { label: "Knowledge Sources", value: "Wikipedia + YouTube + MIT OCW" },
            { label: "GNN Architecture", value: "Graph Convolutional Network" },
            { label: "Adaptation", value: "Real-time mastery tracking" },
          ].map((stat) => (
            <div key={stat.label} className="text-center p-4 rounded-xl"
              style={{ background: "hsl(222,23%,9%)", border: "1px solid hsl(220,18%,15%)" }}>
              <div className="text-xs text-muted-foreground mb-1">{stat.label}</div>
              <div className="text-xs font-medium" style={{ color: "hsl(38,90%,70%)" }}>{stat.value}</div>
            </div>
          ))}
        </motion.div>
      </motion.div>
    </div>
  );
}