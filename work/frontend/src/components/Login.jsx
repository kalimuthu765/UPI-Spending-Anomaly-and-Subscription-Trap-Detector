import { useState } from "react";
import { api, setToken } from "../api.js";

const POINTS = [
  { t: "Spending by shop", d: "See exactly how much went to each shop and category." },
  { t: "Unusual payments", d: "Spot the payment that doesn't fit your normal pattern, month by month." },
  { t: "Subscriptions", d: "Find plans that keep renewing, and what they cost you a year." },
];

export default function Login({ onAuthed }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const result =
        mode === "login" ? await api.login(email, password) : await api.register(email, password, displayName);
      setToken(result.access_token);
      onAuthed();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <section className="brand" aria-label="About Spend Ledger">
        <span className="logo big" aria-hidden="true">₹</span>
        <h1>Spend Ledger</h1>
        <p className="lede">
          Upload a bank or UPI statement. Your own model learns only from your own payments, so "unusual" always
          means unusual <em>for you</em>.
        </p>
        <ul className="points">
          {POINTS.map((p) => (
            <li key={p.t}><strong>{p.t}</strong><span>{p.d}</span></li>
          ))}
        </ul>
      </section>

      <section className="auth-card">
        <h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2>
        <p className="muted">{mode === "login" ? "Log in to see your ledger." : "It takes a few seconds."}</p>
        <div className="seg" role="tablist">
          <button role="tab" aria-selected={mode === "login"} className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setError(""); }}>Log in</button>
          <button role="tab" aria-selected={mode === "register"} className={mode === "register" ? "active" : ""} onClick={() => { setMode("register"); setError(""); }}>Create account</button>
        </div>
        <form onSubmit={submit}>
          {mode === "register" && (
            <label>Name<input value={displayName} onChange={(e) => setDisplayName(e.target.value)} autoComplete="name" placeholder="Your name" /></label>
          )}
          <label>Email<input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" placeholder="you@example.com" /></label>
          <label>
            Password
            <span className="pw">
              <input type={show ? "text" : "password"} required minLength={mode === "register" ? 6 : undefined} value={password}
                onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} />
              <button type="button" className="quiet" onClick={() => setShow(!show)} aria-pressed={show}>{show ? "Hide" : "Show"}</button>
            </span>
          </label>
          {error && <p className="error" role="alert">{error}</p>}
          <button type="submit" className="primary wide" disabled={busy}>
            {busy ? "One moment…" : mode === "login" ? "Log in" : "Create account"}
          </button>
        </form>
      </section>
    </main>
  );
}
