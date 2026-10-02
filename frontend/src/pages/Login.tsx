/**
 * Sign-in — the console's front door.
 *
 * One account owns everything downstream (missions, runs, artifacts), so the
 * page states what the user is signing in to and nothing else: the brand, the
 * two fields, and an honest error when credentials are refused.
 */

import { useEffect, useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useSession } from "../hooks/useSession";
import { useTheme } from "../hooks/useTheme";
import { friendlyMessage } from "../lib/errors";
import { Icon } from "../components/ui/Icon";

type Mode = "signin" | "create";

export default function Login() {
  const navigate = useNavigate();
  const { status, signIn, register } = useSession();
  const { resolved, toggle: toggleTheme } = useTheme();
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const emailRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    emailRef.current?.focus();
  }, []);

  // Already signed in (or just signed in): the console is the destination.
  useEffect(() => {
    if (status === "signed-in") navigate("/", { replace: true });
  }, [status, navigate]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "create") await register(email.trim(), password);
      else await signIn(email.trim(), password);
      navigate("/", { replace: true });
    } catch (err) {
      setError(friendlyMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const canSubmit = email.trim().length >= 3 && password.length >= (mode === "create" ? 8 : 1) && !busy;

  return (
    <div className="login">
      <button
        className="icon-btn login-theme"
        onClick={toggleTheme}
        title={resolved === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        aria-label={resolved === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      >
        <Icon name={resolved === "dark" ? "sun" : "moon"} />
      </button>
      <div className="login-panel">
        <div className="login-brand">
          <img src="/strata-mark.png" alt="" className="login-mark" />
          <img src="/strata-word.png" alt="STRATA" className="login-word" />
        </div>
        <p className="login-tagline">Drone photogrammetry console</p>

        <form className="login-form" onSubmit={submit}>
          <h1 className="login-title">{mode === "signin" ? "Sign in" : "Create your account"}</h1>
          <p className="login-sub">
            {mode === "signin"
              ? "Your missions, reconstructions and reports live behind this sign-in."
              : "New accounts start with survey access to missions and runs."}
          </p>

          <label className="login-field">
            <span>Email</span>
            <input
              ref={emailRef}
              type="email"
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@survey.org"
              required
            />
          </label>

          <label className="login-field">
            <span>Password</span>
            <input
              type="password"
              autoComplete={mode === "create" ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "create" ? "At least 8 characters" : "••••••••"}
              required
            />
          </label>

          {error && (
            <div className="login-error" role="alert">
              <Icon name="warning" /> <span>{error}</span>
            </div>
          )}

          <button className="btn btn-primary login-submit" type="submit" disabled={!canSubmit}>
            {busy ? "Signing in…" : mode === "signin" ? "Sign in" : "Create account"}
            {!busy && <Icon name="arrow-up-right" />}
          </button>

          <button
            type="button"
            className="login-switch"
            onClick={() => {
              setMode((m) => (m === "signin" ? "create" : "signin"));
              setError(null);
            }}
          >
            {mode === "signin" ? "Need an account? Create one" : "Already have an account? Sign in"}
          </button>
        </form>
      </div>

      <aside className="login-aside" aria-hidden="true">
        <div className="login-aside-inner">
          <h2>From flight footage to a measurable model</h2>
          <ul>
            <li>Frames extracted and conditioned for reconstruction</li>
            <li>Sparse, dense and meshed geometry from real imagery</li>
            <li>Metric accuracy checked against ground truth where available</li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
