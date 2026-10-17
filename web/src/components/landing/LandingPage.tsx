import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import EnhancedBackgroundPaths from "@/components/ui/modern-background-paths";
import { api } from "@/lib/api";
import { useAppState } from "@/lib/store";

export function LandingPage() {
  const { setSession } = useAppState();
  const [goalInput, setGoalInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goalInput.trim()) return;

    setIsGenerating(true);
    try {
      const data = await api.buildPath(goalInput);
      setSession(data.session_id, goalInput, data);
    } catch (err) {
      alert("Failed to build path: " + (err as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <>
      <EnhancedBackgroundPaths title="Learning Journey Builder" />

      <div className="relative z-10 container mx-auto px-4 md:px-6 lg:px-8 flex min-h-screen items-center justify-center">
        <div className="w-full max-w-3xl space-y-6">
          <div className="text-center">
            <h1 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-300 dark:from-slate-50 dark:to-white mb-4 tracking-tighter">
              Agentic AI<br/>
              <span className="block">Learning Journey Builder</span>
            </h1>
            <p className="text-lg md:text-xl text-slate-400 dark:text-slate-500 max-w-2xl mx-auto">
              Transform any learning goal into an intelligent, adaptive journey through AI-generated prerequisite knowledge graphs.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <label htmlFor="goal-input" className="block text-sm font-medium mb-1 text-slate-300 dark:text-slate-400">
                What do you want to master?
              </label>
              <Input
                id="goal-input"
                value={goalInput}
                onChange={(e) => setGoalInput(e.target.value)}
                placeholder="e.g. Master Quantum Computing, Learn Rust Systems Programming..."
                className="rounded-xl border border-subtle bg-surface-1/80 backdrop-blur-sm px-4 py-3 text-lg focus-visible:ring-2 focus-visible:ring-primary/20 disabled:opacity-50"
                disabled={isGenerating}
              />
            </div>

            <Button
              type="submit"
              size="lg"
              variant="default"
              className="w-full rounded-xl px-6 py-3 text-lg font-medium transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
              disabled={isGenerating || !goalInput.trim()}
            >
              {isGenerating ? (
                <>
                  <span className="mr-2">Analyzing</span>
                  <span className="inline-block h-3 w-3 animate-spin border-2 border-primary rounded-full border-transparent border-t-primary"></span>
                </>
              ) : (
                <>
                  Build My Learning Journey
                  <span className="ml-2">→</span>
                </>
              )}
            </Button>
          </form>

          <div className="text-center text-slate-500 dark:text-slate-400 text-sm mt-8">
            Powered by Gemini AI • Knowledge Graph Technology • Adaptive Learning
          </div>
        </div>
      </div>
    </>
  );
}