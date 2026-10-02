import { useEffect, useRef, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { api } from "../../lib/api";
import { useRuns } from "../../hooks/useRuns";
import { useSession } from "../../hooks/useSession";
import { useTheme } from "../../hooks/useTheme";
import { Icon, type IconName } from "../ui/Icon";

const NAV: { to: string; label: string; icon: IconName; end?: boolean }[] = [
  { to: "/", label: "Dashboard", icon: "dashboard", end: true },
  { to: "/missions", label: "Missions", icon: "missions" },
  { to: "/viewer", label: "Viewer", icon: "cube" },
  { to: "/analysis", label: "Analysis", icon: "chart" },
  { to: "/reports", label: "Reports", icon: "report" },
];

export default function TopBar() {
  const [health, setHealth] = useState<"ok" | "down" | "loading">("loading");
  const { runs, selected, selectRun } = useRuns();
  const { user, signOut } = useSession();
  const { resolved, toggle: toggleTheme } = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api
      .getHealth()
      .then(() => setHealth("ok"))
      .catch(() => setHealth("down"));
  }, []);

  // Close the account menu on any click outside it.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [menuOpen]);

  const doSignOut = () => {
    setMenuOpen(false);
    void signOut();
  };

  return (
    <header className="topbar">
      <NavLink to="/" className="brand" aria-label="STRATA home">
        <img src="/strata-mark.png" alt="" className="brand-logo" />
        <img src="/strata-word.png" alt="STRATA" className="brand-word" />
      </NavLink>

      <nav className="topnav" aria-label="Primary">
        {/* Each link also carries its own name: the visible label is hidden at
            narrow widths, and without aria-label/title the nav becomes five
            unnamed links (no screen-reader name, no hover tooltip). */}
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            aria-label={n.label}
            title={n.label}
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            <span className="nav-icon">
              <Icon name={n.icon} />
            </span>
            <span className="nav-label">{n.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="topbar-right">
        {runs.length > 0 && (
          <label className="run-picker">
            <span className="run-picker-label">Run</span>
            <select
              className="run-select"
              value={selected?.run_id ?? ""}
              onChange={(e) => selectRun(e.target.value)}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.mission ?? r.dataset ?? "Run"} — {r.status ?? "?"} ({r.run_id.slice(-6)})
                </option>
              ))}
            </select>
          </label>
        )}

        <span className={`topbar-status ${health}`} title={`Backend ${health}`}>
          <span className="status-dot" />
        </span>

        <button
          className="icon-btn"
          onClick={toggleTheme}
          title={resolved === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          aria-label={resolved === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          <Icon name={resolved === "dark" ? "sun" : "moon"} />
        </button>

        <div className="account" ref={menuRef}>
          <button
            className="account-btn"
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <span className="user-avatar">{(user?.email ?? "?").charAt(0).toUpperCase()}</span>
            <span className="user-email">{user?.email ?? "Signed out"}</span>
          </button>
          {menuOpen && (
            <div className="account-menu" role="menu">
              <div className="account-menu-head">
                <strong>{user?.email}</strong>
                <span className="account-role">{user?.role ?? "viewer"}</span>
              </div>
              <button
                role="menuitem"
                className="account-menu-item"
                onClick={() => {
                  setMenuOpen(false);
                  navigate("/settings");
                }}
              >
                <Icon name="gear" /> Settings
              </button>
              <button role="menuitem" className="account-menu-item danger" onClick={doSignOut}>
                <Icon name="x" /> Sign out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
