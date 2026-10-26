import { LandingPage } from "@/components/landing/LandingPage";
import { LearningJourney } from "@/components/dashboard/LearningJourney";
import { useAppState } from "@/lib/store";

export function App() {
  const { sessionId } = useAppState();
  return (
    <div className="dark min-h-screen bg-background text-foreground font-sans antialiased">
      {sessionId ? <LearningJourney /> : <LandingPage />}
    </div>
  );
}
