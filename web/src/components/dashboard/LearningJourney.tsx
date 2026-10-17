import { useState, useEffect, FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAppState } from "@/lib/store";
import { api } from "@/lib/api";
import { GraphVisualizer } from "@/components/graph/GraphVisualizer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import confetti from "canvas-confetti";

export function LearningJourney() {
  const { sessionId, goal, graphData, currentLesson, isGenerating, setCurrentLesson, setIsGenerating, updateGraph, reset } = useAppState();
  const [activeTab, setActiveTab] = useState("journey");
  const [selectedAnswers, setSelectedAnswers] = useState<number[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedConcept, setSelectedConcept] = useState<any | null>(null);
  const [showConceptDetail, setShowConceptDetail] = useState(false);

  const loadNextLesson = async () => {
    if (!sessionId) return;
    try {
      const lesson = await api.getNextLesson(sessionId);
      setCurrentLesson(lesson);
      setSelectedAnswers([]);
      if (lesson.done) {
        confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
      }
    } catch (err) {
      console.error("Failed to load lesson:", err);
    } finally {
      setIsGenerating(false);
    }
  };

  const submitQuiz = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionId || !currentLesson?.node_id) return;

    setIsSubmitting(true);
    try {
      const result = await api.submitAnswer(sessionId, currentLesson.node_id, selectedAnswers);

      // Update graph progress natively
      const updatedGraph = await api.getGraph(sessionId);
      updateGraph(updatedGraph);

      // Show feedback
      alert(`Score: ${Math.round(result.score * 100)}%\n\n${result.feedback.join('\n')}`);

      if (result.passed) setCurrentLesson(null); // Clear to load next
      if (result.goal_complete) {
        confetti({ particleCount: 150, spread: 80, origin: { y: 0.6 } });
        setTimeout(() => {
          // Reset after celebration
          reset();
        }, 3000);
      }
    } catch (err) {
      console.error("Submit failed:", err);
      alert("Submission failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleConceptSelect = (concept: any) => {
    setSelectedConcept(concept);
    setShowConceptDetail(true);
  };

  // Calculate progress metrics
  useEffect(() => {
    if (graphData && graphData.progress) {
      const nodes = graphData.nodes;
      const progress = graphData.progress;
      const masteredCount = Object.values(progress.node_status).filter(s => s === "mastered").length;
      const percentComplete = Math.round((masteredCount / nodes.length) * 100);
      // Update document title with progress
      document.title = `${percentComplete}% • ${goal} • Learning Journey`;
    }
  }, [graphData, goal]);

  if (!sessionId || !graphData) {
    // This shouldn't happen in normal flow, but just in case
    return null;
  }

  const nodes = graphData.nodes;
  const progress = graphData.progress;

  const masteredCount = Object.values(progress.node_status).filter(s => s === "mastered").length;
  const percentComplete = Math.round((masteredCount / nodes.length) * 100);
  const availableCount = Object.values(progress.node_status).filter(s => s === "available").length;
  const inProgressCount = Object.values(progress.node_status).filter(s => s === "in_progress").length;
  const lockedCount = Object.values(progress.node_status).filter(s => s === "locked").length;

  return (
    <>
      {/* Header */}
      <header className="border-b border-subtle bg-surface-1/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 md:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 flex items-center justify-center bg-surface-2 rounded-md">
              <span className="text-primary-custom">🧭</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-primary-custom tracking-tight">
                {goal}
              </h1>
              <p className="text-sm text-secondary-custom">
                {masteredCount} of {nodes.length} concepts mastered
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Badge
                variant={masteredCount === nodes.length ? "success" : "outline"}
                className="px-3 py-1 text-xs"
              >
                {percentComplete}% Complete
              </Badge>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (confirm("Reset learning journey?")) {
                  reset();
                }
              }}
            >
              Reset Journey
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 md:px-6 lg:px-8 py-6">
        <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">

          {/* Left Panel: Interactive Knowledge Graph */}
          <div className="lg:col-span-1">
            <Card className="h-full">
              <CardHeader className="pb-4">
                <h2 className="text-xl font-semibold text-primary-custom mb-2">
                  Knowledge Graph
                </h2>
                <p className="text-sm text-secondary-custom">
                  Prerequisite relationships and learning dependencies
                </p>
              </CardHeader>
              <CardContent className="p-0">
                <div className="h-[500px] w-full">
                  <GraphVisualizer
                    graphData={graphData}
                    onNodeClick={handleConceptSelect}
                  />
                </div>
                {showConceptDetail && selectedConcept && (
                  <AnimatePresence mode="popLayout">
                    <motion.div
                      key={`detail-${selectedConcept.id}`}
                      initial={{ x: 20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      exit={{ x: -20, opacity: 0 }}
                      className="absolute right-0 top-0 w-[320px] h-full bg-surface-2 border-l border-subtle"
                      style={{ zIndex: 10 }}
                    >
                      <div className="p-4 flex flex-col h-full space-y-4">
                        <div className="flex justify-between items-start mb-2">
                          <h3 className="text-lg font-bold text-primary-custom">{selectedConcept.title}</h3>
                          <Badge
                            variant={progress.node_status[selectedConcept.id] === "mastered" ? "success" :
                                     progress.node_status[selectedConcept.id] === "in_progress" ? "outline" :
                                     progress.node_status[selectedConcept.id] === "available" ? "secondary" : "destructive"}
                            className="px-2 py-0.5 text-xs"
                          >
                            {progress.node_status[selectedConcept.id]}
                          </Badge>
                        </div>

                        <div className="flex-1 min-h-0">
                          <p className="text-sm text-secondary-custom leading-relaxed mb-2">
                            {selectedConcept.description}
                          </p>

                          {selectedConcept.source_url && (
                            <div className="mt-2 flex items-center space-x-2 text-sm">
                              <a
                                href={selectedConcept.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="flex items-center gap-1 text-primary-custom hover:underline"
                              >
                                <span className="w-3 h-3 flex items-center justify-center bg-primary/20 rounded">
                                  🔗
                                </span>
                                View Source Material
                              </a>
                            </div>
                          )}
                        </div>

                        <div className="mt-4 pt-3 border-t border-subtle">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span>Difficulty:</span>
                            <span className="font-medium">{selectedConcept.difficulty}/5</span>
                          </div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span>Mastery:</span>
                            <span className="font-medium">
                              {progress.node_scores[selectedConcept.id] !== undefined ?
                                `${Math.round(progress.node_scores[selectedConcept.id] * 100)}%` :
                                "Not attempted"}
                            </span>
                          </div>

                          <div className="mt-3">
                            <Button
                              variant="outline"
                              size="sm"
                              className="w-full"
                              onClick={() => setShowConceptDetail(false)}
                            >
                              Close Details
                            </Button>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  </AnimatePresence>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Panel: Learning Journey & Tutor */}
          <div className="lg:col-span-2">
            <div className="space-y-6">

              {/* Progress Overview */}
              <Card className="h-[120px]">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-primary-custom">Learning Progress</h2>
                    <div className="flex items-center gap-2 text-sm">
                      <div className="w-8 h-8 flex items-center justify-center bg-primary/20 rounded-lg">
                        <span className="text-primary-custom">📊</span>
                      </div>
                      <span className="font-medium">{masteredCount}/{nodes.length}</span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="grid gap-4 p-4">
                    <div className="flex flex-col">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Completed:</span>
                        <span className="font-medium text-success-foreground">{masteredCount}</span>
                      </div>
                      <div className="w-full bg-border/20 h-1.5 rounded">
                        <div
                          className={`h-full bg-success-foreground rounded transition-all duration-500 w-[${masteredCount / nodes.length * 100}%]`}
                        ></div>
                      </div>
                    </div>
                    <div className="flex flex-col">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Available:</span>
                        <span className="font-medium text-accent-foreground">{availableCount}</span>
                      </div>
                      <div className="w-full bg-border/20 h-1.5 rounded">
                        <div
                          className={`h-full bg-accent-foreground rounded transition-all duration-500 w-[${availableCount / nodes.length * 100}%]`}
                        ></div>
                      </div>
                    </div>
                    <div className="flex flex-col">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>In Progress:</span>
                        <span className="font-medium text-warning-foreground">{inProgressCount}</span>
                      </div>
                      <div className="w-full bg-border/20 h-1.5 rounded">
                        <div
                          className={`h-full bg-warning-foreground rounded transition-all duration-500 w-[${inProgressCount / nodes.length * 100}%]`}
                        ></div>
                      </div>
                    </div>
                    <div className="flex flex-col">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Locked:</span>
                        <span className="font-medium text-muted-foreground">{lockedCount}</span>
                      </div>
                      <div className="w-full bg-border/20 h-1.5 rounded">
                        <div
                          className={`h-full bg-muted-foreground rounded transition-all duration-500 w-[${lockedCount / nodes.length * 100}%]`}
                        ></div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Main Content Tabs */}
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-2 max-w-[400px]">
                  <TabsTrigger value="journey">Learning Journey</TabsTrigger>
                  <TabsTrigger value="graph">Knowledge Explorer</TabsTrigger>
                </TabsList>

                <TabsContent value="journey" className="flex-1 mt-6">
                  <AnimatePresence mode="wait">
                    {currentLesson ? (
                      <motion.div
                        key="lesson-view"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="border-primary/20 shadow-lg"
                      >
                        <Card className="border-primary/20 shadow-lg">
                          <CardHeader className="bg-primary/5 border-b border-border/50 pb-6 rounded-t-xl">
                            <div className="flex justify-between items-start text-primary-custom">
                              <div>
                                <Badge variant="glow" className="mb-2">Current Focus</Badge>
                                <CardTitle className="text-2xl">{currentLesson.node_title}</CardTitle>
                              </div>
                              <Badge
                                variant="outline"
                                className="bg-background/80"
                              >
                                Difficulty: {currentLesson.difficulty}/5
                              </Badge>
                            </div>
                            {currentLesson.source_url && (
                              <a
                                href={currentLesson.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-sm flex items-center text-secondary-custom hover:text-primary-custom transition-colors mt-2"
                              >
                                <span className="w-3 h-3 flex items-center justify-center bg-primary/20 rounded mr-1">
                                  📚
                                </span>
                                External Resource
                              </a>
                            )}
                          </CardHeader>
                          <CardContent className="pt-6 space-y-8">
                            <div className="prose prose-sm md:prose-base dark:prose-invert max-w-none text-primary-custom/90 leading-relaxed">
                              {currentLesson.lesson?.split('\n').map((para, i) => (
                                <p key={i} className="mb-4 last:mb-0">
                                  {para}
                                </p>
                              ))}
                            </div>

                            <Separator className="my-6" />

                            <div>
                              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                Mastery Check <span className="text-xs font-normal text-secondary-custom">(Pass ≥ 70%)</span>
                              </h3>
                              <form onSubmit={submitQuiz} className="space-y-6">
                                {currentLesson.quiz?.map((q, idx) => (
                                  <div key={idx} className="space-y-4 bg-surface-1/50 p-4 rounded-lg border border-border/50">
                                    <p className="font-medium text-sm">
                                      {idx + 1}. {q.question}
                                    </p>
                                    <RadioGroup
                                      onValueChange={(val) => {
                                        const newAns = [...selectedAnswers];
                                        newAns[idx] = parseInt(val);
                                        setSelectedAnswers(newAns);
                                      }}
                                    >
                                      {q.options.map((opt, optIdx) => (
                                        <div key={optIdx} className="flex items-center space-x-2">
                                          <RadioGroupItem
                                            value={optIdx.toString()}
                                            id={`q${idx}-o${optIdx}`}
                                          />
                                          <label
                                            htmlFor={`q${idx}-o${optIdx}`}
                                            className="text-sm font-normal cursor-pointer leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-50"
                                          >
                                            {opt}
                                          </label>
                                        </div>
                                      ))}
                                    </RadioGroup>
                                  </div>
                                ))}

                                <Button
                                  type="submit"
                                  disabled={isSubmitting || selectedAnswers.length !== currentLesson.quiz?.length}
                                  className="w-full sm:w-auto"
                                >
                                  {isSubmitting ? "Evaluating..." : "Submit Answers"}
                                </Button>
                              </form>
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="ready-state"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="h-[400px] flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl bg-surface-1/50 text-center"
                      >
                        <div className="flex items-center justify-center mb-4">
                          <div className="w-12 h-12 flex items-center justify-center bg-primary/20 rounded-xl">
                            <span className="text-primary-custom">🎯</span>
                          </div>
                        </div>
                        <h3 className="text-xl font-semibold mb-2 text-primary-custom">
                          Ready to Learn
                        </h3>
                        <p className="text-secondary-custom max-w-sm mb-6">
                          Your personalized micro-lessons are ready. We adapt the content dynamically based on your mastery and the knowledge graph trajectory.
                        </p>
                        <Button
                          size="lg"
                          onClick={loadNextLesson}
                          disabled={isSubmitting || isGenerating}
                        >
                          Begin Learning
                        </Button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </TabsContent>

                <TabsContent value="graph" className="flex-1 mt-6 h-[600px]">
                  <Card className="h-full">
                    <CardHeader className="pb-4">
                      <h2 className="text-xl font-semibold text-primary-custom mb-2">
                        Knowledge Explorer
                      </h2>
                      <p className="text-sm text-secondary-custom">
                        Explore concept relationships and dependencies
                      </p>
                    </CardHeader>
                    <CardContent className="p-0">
                      <div className="h-full w-full">
                        <GraphVisualizer
                          graphData={graphData}
                          onNodeClick={handleConceptSelect}
                        />
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </main>

      {/* Floating Action Button for New Journey */}
      <div className="fixed bottom-6 right-6 z-50">
        <Button
          variant="default"
          size="lg"
          className="w-14 h-14 flex items-center justify-center rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
          onClick={() => {
            if (confirm("Start a new learning journey?")) {
              reset();
            }
          }}
        >
          <span className="text-primary-custom">+</span>
        </Button>
      </div>
    </>
  );
}