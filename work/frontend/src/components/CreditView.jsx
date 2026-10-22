import { useMemo, useState } from "react";
import { rupees } from "../format.js";

const PAGE = 25;

export default function CreditView({ credit }) {
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);

  const q = query.trim().toLowerCase();
  const sources = useMemo(
    () => credit.sources.filter((s) => !q || s.source.toLowerCase().includes(q)), [credit, q]);
  const rows = useMemo(
    () => credit.rows.filter((r) => !q || r.source.toLowerCase().includes(q) || r.kind.toLowerCase().includes(q)),
    [credit, q]);
  const filteredTotal = rows.reduce((s, r) => s + r.amount, 0);
  const maxMonth = Math.max(1, ...credit.by_month.map((m) => m.total));

  if (!credit.has_credits) {
    return (
      <div className="credit-empty">
        <h2>Credit: money that came in</h2>
        <p>No money received was found in your statements yet.</p>
        <p className="muted">
          Credits show up when a statement lists them (for example "Received from", a Credit type, or a Credit
          column). If you uploaded before this view existed, upload the same file again. Your spending will not be
          counted twice.
        </p>
      </div>
    );
  }

  return (
    <div className="credit">
      <h2>Credit: money that came in</h2>
      <p className="muted">Everything you received, kept apart from your spending.</p>

      <dl className="facts credit-facts">
        <div className="tile t-in"><dt>Total received</dt><dd>{rupees(credit.total)}</dd></div>
        <div className="tile t-credit"><dt>Credits</dt><dd>{credit.other_count}<span>{rupees(credit.other_total)}</span></dd></div>
        <div className="tile t-refund"><dt>Refunds</dt><dd>{credit.refund_count}<span>{rupees(credit.refund_total)}</span></dd></div>
        <div className="tile t-src"><dt>Sources</dt><dd>{credit.sources.length}</dd></div>
      </dl>

      <h3 className="sub-h">Month by month</h3>
      <ul className="periods credit-months">
        {credit.by_month.map((m) => (
          <li key={m.label}>
            <div className="row"><strong>{m.label}</strong><span className="amt">{rupees(m.total)}</span></div>
            <div className="bar in"><span style={{ width: `${(m.total / maxMonth) * 100}%` }} /></div>
            <span className="muted">
              {m.count} {m.count === 1 ? "credit" : "credits"}
              {m.refund_total > 0 && <> · refunds {rupees(m.refund_total)}</>}
              {m.top_sources.length > 0 && <> · mostly {m.top_sources.map((s) => `${s.source} ${rupees(s.total)}`).join(", ")}</>}
            </span>
          </li>
        ))}
      </ul>

      <h3 className="sub-h">Where it came from</h3>
      <div className="filters">
        <input type="search" placeholder="Search a source, for example your company" value={query}
          onChange={(e) => { setQuery(e.target.value); setShown(PAGE); }} aria-label="Search credits" />
        <span className="muted">{rows.length} credits, {rupees(filteredTotal)}</span>
      </div>

      <div className="scroll">
        <table className="sources-table">
          <thead><tr><th>Source</th><th className="num">Times</th><th className="num">Total</th><th>Last received</th></tr></thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s.source}>
                <td className="svc">{s.source}{s.refunds > 0 && <span className="chip-refund">refund</span>}</td>
                <td className="num" data-label="Times">{s.count}</td>
                <td className="num amt" data-label="Total">{rupees(s.total)}</td>
                <td data-label="Last received">{s.last}</td>
              </tr>
            ))}
            {sources.length === 0 && <tr><td colSpan="4" className="muted">No source matches.</td></tr>}
          </tbody>
        </table>
      </div>

      <h3 className="sub-h">All credits</h3>
      <div className="scroll">
        <table>
          <thead><tr><th>Date</th><th>From</th><th>Type</th><th className="num">Amount</th></tr></thead>
          <tbody>
            {rows.slice(0, shown).map((r, i) => (
              <tr key={i}>
                <td>{r.date}</td><td>{r.source}</td>
                <td><span className={`kind-chip ${r.kind === "Refund" ? "refund" : ""}`}>{r.kind}</span></td>
                <td className="num amt">{rupees(r.amount)}</td>
              </tr>
            ))}
            {rows.length === 0 && <tr><td colSpan="4" className="muted">No credits match.</td></tr>}
          </tbody>
        </table>
      </div>
      {shown < rows.length && (
        <button className="quiet" onClick={() => setShown(shown + PAGE)}>Show {Math.min(PAGE, rows.length - shown)} more</button>
      )}
    </div>
  );
}
