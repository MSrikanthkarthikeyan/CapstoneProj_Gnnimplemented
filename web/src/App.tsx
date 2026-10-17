import React from "react";
import { LandingPage } from "@/components/landing/LandingPage";
import { LearningJourney } from "@/components/dashboard/LearningJourney";
import { useAppState } from "@/lib/store";

export function App() {
  const { sessionId } = useAppState();

  return (
    <div className="min-h-screen bg-background text-foreground font-sans antialiased selection:bg-primary/20">
      {sessionId ? <LearningJourney /> : <LandingPage />}
    </div>
  );
}
