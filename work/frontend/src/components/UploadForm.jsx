import { useRef, useState } from "react";
import { api } from "../api.js";

const FORMATS = ["PDF", "Excel", "CSV", "JSON", "Word"];

export default function UploadForm({ onUploaded, compact }) {
  const input = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [over, setOver] = useState(false);

  async function send(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      onUploaded(await api.uploadHistory(file), file.name);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  return (
    <div>
      <div
        className={`drop ${over ? "over" : ""} ${compact ? "compact" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); send(e.dataTransfer.files[0]); }}
      >
        <div>
          <strong>{busy ? "Reading your statement…" : compact ? "Add more transactions" : "Drop a statement here"}</strong>
          <p className="muted">
            Drag a file here, or choose one. New payments are added to your history and your own model learns from them.
          </p>
          <div className="formats" aria-label="Accepted formats">
            {FORMATS.map((f) => <span key={f} className="fmt">{f}</span>)}
          </div>
        </div>
        <button className="primary" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? <><span className="spinner" aria-hidden="true" /> Analysing…</> : "Choose file"}
        </button>
        <input
          ref={input} type="file" hidden
          accept=".csv,.pdf,.xlsx,.xls,.json,.docx"
          onChange={(e) => send(e.target.files[0])}
        />
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}
