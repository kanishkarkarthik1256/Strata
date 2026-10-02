import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Missions from "./pages/Missions";
import NewMission from "./pages/NewMission";
import Processing from "./pages/Processing";
import Viewer from "./pages/Viewer";
import Analysis from "./pages/Analysis";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import Login from "./pages/Login";
import TopBar from "./components/layout/TopBar";
import ErrorBoundary from "./components/ui/ErrorBoundary";
import { RunsProvider } from "./hooks/useRuns";
import { SessionProvider, useSession } from "./hooks/useSession";
import { ThemeProvider } from "./hooks/useTheme";

/** Route table shared by the signed-in console. */
const ROUTES = [
  { path: "/", element: <Dashboard /> },
  { path: "/missions", element: <Missions /> },
  { path: "/new-mission", element: <NewMission /> },
  { path: "/processing/:jobId", element: <Processing /> },
  { path: "/viewer", element: <Viewer /> },
  { path: "/analysis", element: <Analysis /> },
  { path: "/reports", element: <Reports /> },
  { path: "/settings", element: <Settings /> },
];

/** The signed-in console: taskbar + routed page. */
function Console() {
  const location = useLocation();

  return (
    <div className="app-shell">
      <TopBar />
      <div className="app-main">
        <div className="app-content">
          <ErrorBoundary>
            {/* Keyed wrapper: each navigation fades its page in, so moving
                between pages reads as one continuous flow. */}
            <div className="page" key={location.pathname}>
              <Routes location={location}>
                {ROUTES.map((r) => (
                  <Route key={r.path} path={r.path} element={r.element} />
                ))}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </div>
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}

function Shell() {
  const { status } = useSession();

  if (status === "checking") {
    return (
      <div className="boot">
        <img src="/strata-mark.png" alt="" className="boot-mark" />
        <p>Restoring session…</p>
      </div>
    );
  }
  if (status !== "signed-in") return <Login />;

  // Runs are only fetched once there is a session to fetch them with.
  return (
    <RunsProvider>
      <Console />
    </RunsProvider>
  );
}

export default function App() {
  return (
    // Theme wraps everything, including the signed-out views — the login
    // screen is themed too, so signing in never flashes a different shell.
    <ThemeProvider>
      <SessionProvider>
        <Shell />
      </SessionProvider>
    </ThemeProvider>
  );
}
