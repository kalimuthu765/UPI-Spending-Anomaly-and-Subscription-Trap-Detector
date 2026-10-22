import { useEffect, useState } from "react";

const KEY = "theme"; // "light" | "dark" | "auto"
function read() { try { return localStorage.getItem(KEY) || "auto"; } catch (_) { return "auto"; } }

export default function ThemeToggle() {
  const [mode, setMode] = useState(read);
  useEffect(() => {
    const root = document.documentElement;
    if (mode === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", mode);
    try { localStorage.setItem(KEY, mode); } catch (_) {}
  }, [mode]);
  const opts = [["light", "Light"], ["auto", "Auto"], ["dark", "Dark"]];
  return (
    <div className="theme-toggle" role="group" aria-label="Colour theme">
      {opts.map(([v, l]) => (
        <button key={v} aria-pressed={mode === v} onClick={() => setMode(v)}>{l}</button>
      ))}
    </div>
  );
}
