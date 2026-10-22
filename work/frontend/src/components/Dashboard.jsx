import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api, clearToken } from "../api.js";
import UploadForm from "./UploadForm.jsx";
import KindPager from "./KindPager.jsx";
import CreditView from "./CreditView.jsx";
import { rupees, day } from "../format.js";

const PAGE = 25;
const times = (m) => (m >= 10 ? Math.round(m) : m);

const CAT_COLORS = { Food: "#e07a3f", Groceries: "#5aa35a", Fuel: "#8b6f47", Travel: "#3f8fd0", Bills: "#8f6fd6",
  Entertainment: "#d6529b", Shopping: "#1f9c8a", Health: "#d65252", Education: "#c9a227", Other: "#8a9a96" };
const catColor = (c) => CAT_COLORS[c] || "#8a9a96";

function Ledger({ rows, labelKey, onPick, active }) {
  const max = Math.max(...rows.map((r) => r.total), 1);
  return (
    <ol className="ledger">
      {rows.map((r) => (
        <li key={r[labelKey]}>
          <button
            className={active === r[labelKey] ? "active" : ""}
            onClick={onPick ? () => onPick(active === r[labelKey] ? null : r[labelKey]) : undefined}
            disabled={!onPick}
            aria-pressed={onPick ? active === r[labelKey] : undefined}
          >
            <span className="name">
              {labelKey === "category" && <i className="dot" style={{ background: catColor(r[labelKey]) }} aria-hidden="true" />}
              {r[labelKey]}
            </span>
            <span className="leader" aria-hidden="true" />
            <span className="amt">{rupees(r.total)}</span>
            <span className="meta">
              {r.count != null ? `${r.count} ${r.count === 1 ? "payment" : "payments"}` : ""}
              {r.average != null && r.count > 1 ? `, about ${rupees(r.average)} each` : ""}
              {r.share_pct != null ? ` · ${r.share_pct}% of spending` : ""}
            </span>
            <span className="bar" style={{ width: `${(r.total / max) * 100}%` }} aria-hidden="true" />
          </button>
        </li>
      ))}
    </ol>
  );
}


function Attention({ summary }) {
  const b = summary.biggest;
  function open() {
    document.getElementById("tab-unusual")?.click();
    document.querySelector(".pager")?.scrollIntoView({ behavior: "smooth" });
  }
  if (!summary.count) {
    return (
      <section className="attention calm">
        <h2>Nothing unusual so far</h2>
        <p className="muted">Every payment fits your usual pattern. We check each new upload against your own history.</p>
      </section>
    );
  }
  return (
    <section className="attention" aria-labelledby="attn-h">
      <h2 id="attn-h">
        {summary.count} {summary.count === 1 ? "payment looks" : "payments look"} unusual for you
      </h2>
      <p className="big">
        <strong>{rupees(b.amount)}</strong> at {b.merchant} on {day(b.timestamp)}
        <span className="mult">
          {b.multiple >= 2 ? `${times(b.multiple)}× your usual payment of ${rupees(summary.typical_payment)}` : "higher than you normally spend there"}
        </span>
      </p>
      <ul className="months">
        {summary.by_month.map((m) => (
          <li key={m.month} className={m.is_latest_month ? "now" : ""}>
            <span>{m.month}{m.is_latest_month ? " (latest)" : ""}</span>
            <span>{m.count} unusual, {rupees(m.total)}</span>
          </li>
        ))}
      </ul>
      <p className="muted">
        Together these are {rupees(summary.total_amount)}, {summary.share_pct}% of everything you spent.
        Often they are planned purchases, so this is only a prompt to check.
      </p>
      <button className="primary" onClick={open}>See the {summary.count === 1 ? "payment" : "payments"}</button>
    </section>
  );
}

function Periods({ periods, granularity }) {
  const max = Math.max(...periods.map((p) => p.total), 1);
  const unit = granularity === "month" ? "Month by month" : granularity === "week" ? "Week by week" : "Day by day";
  return (
    <section>
      <h2>{unit}: how much went out</h2>
      <ul className="periods">
        {periods.map((p) => (
          <li key={p.label}>
            <div className="row">
              <strong>{p.label}{p.partial ? " (so far)" : ""}</strong>
              <span className="amt">{rupees(p.total)}</span>
            </div>
            <div className="bar"><span style={{ width: `${(100 * p.total) / max}%` }} /></div>
            <p className="muted">
              {p.count} payments · mostly {p.top_shops.map((t) => `${t.merchant} ${rupees(t.amount)}`).join(", ")}
            </p>
            {p.unusual_count > 0 && (
              <p className="period-flag">{p.unusual_count} unusual, {rupees(p.unusual_total)} of this</p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function Pager({ anomalies, shops, categories, subs, subSummary, activeShop, onPickShop }) {
  const [active, setActive] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const track = useRef(null);
  const panels = useRef([]);
  const target = useRef(null);      // index we are smooth-scrolling to (ignore scroll events until we arrive)
  const tablist = useRef(null);
  const tabs = [
    { id: "shops", label: "Spending by shop", count: shops.length },
    { id: "unusual", label: "Worth a look (Anomaly)", count: anomalies.length },
    { id: "subs", label: "Subscriptions", count: subs.length },
  ];

  // keep the tab bar in step with swipes / scrolling
  function onScroll() {
    const t = track.current;
    if (!t) return;
    const i = Math.round(t.scrollLeft / t.clientWidth);
    if (target.current !== null) {
      if (Math.abs(t.scrollLeft - target.current * t.clientWidth) < 2) target.current = null;
      return;
    }
    if (i !== active) setActive(i);
  }
  function go(i) {
    const t = track.current;
    if (!t) return;
    const n = Math.max(0, Math.min(tabs.length - 1, i));
    target.current = n;
    t.scrollTo({ left: n * t.clientWidth, behavior: "smooth" });
    setActive(n);
    setTimeout(() => { target.current = null; }, 700);   // safety: never stay locked
  }

  // on a narrow screen the tab bar scrolls too; keep the active tab visible
  useEffect(() => {
    const bar = tablist.current;
    const el = bar?.children[active];
    if (bar && el) bar.scrollTo({ left: el.offsetLeft - 8, behavior: "smooth" });
  }, [active]);
  function onKey(e) {
    if (e.key === "ArrowRight") { e.preventDefault(); go(active + 1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); go(active - 1); }
  }

  // the track is only as tall as the panel you are looking at
  useLayoutEffect(() => {
    const t = track.current;
    const el = panels.current[active];
    if (!t || !el) return;
    const fit = () => { t.style.height = `${el.offsetHeight}px`; };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [active, showAll, anomalies, shops, subs]);

  const list = showAll ? anomalies : anomalies.slice(0, 12);
  const grouped = Object.entries(list.reduce((g, a) => ((g[a.month] ||= []).push(a), g), {}));

  return (
    <section className="pager" aria-label="Details">
      <div className="pager-bar">
        <div className="tablist" role="tablist" aria-label="Choose a view" onKeyDown={onKey} ref={tablist}>
          {tabs.map((t, i) => (
            <button key={t.id} role="tab" id={`tab-${t.id}`} aria-selected={active === i} aria-controls={`panel-${t.id}`}
              tabIndex={active === i ? 0 : -1} className={active === i ? "active" : ""} onClick={() => go(i)}>
              {t.label} <span className="count">{t.count}</span>
            </button>
          ))}
        </div>
        <div className="arrows">
          <button className="quiet" onClick={() => go(active - 1)} disabled={active === 0} aria-label="Previous view">←</button>
          <button className="quiet" onClick={() => go(active + 1)} disabled={active === tabs.length - 1} aria-label="Next view">→</button>
        </div>
      </div>

      <div className="track" ref={track} onScroll={onScroll}>
        <div className="panel" role="tabpanel" id="panel-shops" aria-labelledby="tab-shops" ref={(el) => (panels.current[0] = el)}>
          <div className="two">
            <div>
              <h2>What you pay at each shop</h2>
              <p className="muted">Select a shop to see only its payments below.</p>
              <Ledger rows={shops} labelKey="merchant" active={activeShop} onPick={onPickShop} />
            </div>
            <div>
              <h2>By category</h2>
              <p className="muted">From your statement; shown as Other when it has none.</p>
              <Ledger rows={categories} labelKey="category" />
            </div>
          </div>
        </div>

        <div className="panel" role="tabpanel" id="panel-unusual" aria-labelledby="tab-unusual" ref={(el) => (panels.current[1] = el)}>
          <h2>Worth a look (Anomaly): payments that look unusual for you</h2>
          <p className="muted">Compared with your own history. Most are fine; this is just a prompt to glance at them.</p>
          {anomalies.length === 0 && <p>Nothing unusual so far.</p>}
          {grouped.map(([month, items]) => (
            <div key={month} className="month-group">
              <h3>
                {month}
                <span className="muted"> · {items.length} unusual, {rupees(items.reduce((t, a) => t + a.amount, 0))}</span>
              </h3>
              <ul className="flags">
                {items.map((a) => (
                  <li key={a.transaction_id}>
                    <div className="row">
                      <strong>{a.merchant}</strong>
                      <span className="amt">{rupees(a.amount)}</span>
                    </div>
                    <span className="muted">{day(a.timestamp)}</span>
                    {a.multiple >= 2 && <span className="chip-mult">{times(a.multiple)}× your usual payment</span>}
                    <p>{a.explanation}</p>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {anomalies.length > 12 && !showAll && (
            <button className="quiet" onClick={() => setShowAll(true)}>Show all {anomalies.length}</button>
          )}
          <p className="swipe-hint muted">Swipe or press → for subscriptions and what keeps going out</p>
        </div>

        <div className="panel" role="tabpanel" id="panel-subs" aria-labelledby="tab-subs" ref={(el) => (panels.current[2] = el)}>
          <h2>Subscriptions and recurring payments</h2>
          {subs.length === 0 ? (
            <p>No repeating payments found yet. They show up once the same shop charges a similar amount a few times.</p>
          ) : (
            <>
              <p className="sub-summary">
                About <strong>{rupees(subSummary.monthly_cost)}</strong> a month, or <strong>{rupees(subSummary.yearly_cost)}</strong> a year.
                You've paid <strong>{rupees(subSummary.total_paid_so_far)}</strong> across these so far.
              </p>
              <div className="scroll">
                <table className="subs-table">
                  <thead><tr><th>Service</th><th>Repeats</th><th className="num">Amount</th><th className="num">Paid so far</th><th>Next expected</th><th>Match</th></tr></thead>
                  <tbody>
                    {subs.map((s) => (
                      <tr key={s.merchant}>
                        <td className="svc">{s.merchant}</td>
                        <td data-label="Repeats"><span style={{ textTransform: "capitalize" }}>{s.cadence}</span></td>
                        <td className="num amt" data-label="Amount">{rupees(s.avg_amount)}</td>
                        <td className="num" data-label="Paid so far">{rupees(s.total_paid)} <span className="muted">({s.times_paid}×)</span></td>
                        <td data-label="Next expected">{s.next_expected_payment ? day(s.next_expected_payment) : "—"}</td>
                        <td data-label="Match"><span className={`tier ${s.confidence.toLowerCase()}`} title="How sure we are this is a repeating payment">{s.confidence}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted">These may be worth reviewing if you no longer use them. "Match" shows how consistent the amount and timing are.</p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

export default function Dashboard({ onLoggedOut }) {
  const [me, setMe] = useState(null);
  const [data, setData] = useState(null);
  const [txns, setTxns] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState(null);
  const [shop, setShop] = useState(null);
  const [query, setQuery] = useState("");
  const [shown, setShown] = useState(PAGE);

  async function load() {
    setLoading(true);
    setError("");
    try {
      setData(await api.dashboard());
      setTxns(await api.recentTransactions(500).catch(() => []));
    } catch (err) {
      if (err.message?.includes("No history")) setData({ empty: true });
      else setError(err.message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { load(); api.me().then(setMe).catch(() => {}); }, []);

  function uploaded(result, name) {
    setNotice({
      name,
      added: result.new_transactions_added,
      credits: result.new_credits_added || 0,
      retrained: result.model_retrained_this_upload,
      spike: result.drift_report?.shift_type === "temporary_spike_suspected",
      changed: result.drift_report?.shift_type === "sustained",
    });
    setShop(null);
    load();
  }

  async function retrainNow() {
    try { await api.retrainNow(); await load(); } catch (_) {}
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return txns.filter((t) => (!shop || t.merchant === shop) &&
      (!q || t.merchant.toLowerCase().includes(q) || (t.category || "").toLowerCase().includes(q)));
  }, [txns, shop, query]);

  const categories = useMemo(() => data && !data.empty
    ? Object.entries(data.spending_by_category).map(([category, total]) => ({ category, total }))
    : [], [data]);

  const filteredTotal = filtered.reduce((s, t) => s + t.amount, 0);

  return (
    <div className="page">
      <header className="top">
        <span className="wordmark"><span className="logo" aria-hidden="true">₹</span>Spend Ledger</span>
        <div className="account">
          {me && (
            <>
              <span className="avatar" aria-hidden="true">{(me.display_name || me.email || "?").trim().charAt(0).toUpperCase()}</span>
              <span className="who">
                {me.display_name && <strong>{me.display_name}</strong>}
                <span className="muted">{me.email}</span>
              </span>
            </>
          )}
          <button className="logout" onClick={() => { clearToken(); onLoggedOut(); }}>Log out</button>
        </div>
      </header>

      {loading && !data && (
        <div className="skeleton" aria-busy="true" aria-label="Loading your ledger">
          <div className="sk sk-hero" />
          <div className="sk-row"><div className="sk sk-tile" /><div className="sk sk-tile" /><div className="sk sk-tile" /><div className="sk sk-tile" /></div>
          <div className="sk sk-block" />
        </div>
      )}
      {error && (
        <div className="error-card" role="alert">
          <strong>We couldn't load your ledger.</strong>
          <p>{error}</p>
          <button className="primary" onClick={() => { setError(""); setLoading(true); load(); }}>Try again</button>
        </div>
      )}

      {data?.empty && (
        <section className="start">
          <ol className="steps">
            <li><b>1</b><span>Upload a statement</span></li>
            <li><b>2</b><span>We learn your normal spending</span></li>
            <li><b>3</b><span>See anomalies, shops and subscriptions</span></li>
          </ol>
          <h1>Start with one statement</h1>
          <p className="lede">
            Upload a Google Pay, PhonePe or bank statement. You'll get your spending by shop, anything unusual for
            you, and the subscriptions that keep renewing.
          </p>
          <UploadForm onUploaded={uploaded} />
        </section>
      )}

      {notice && (
        <div className={`notice ${notice.spike ? "watch" : ""}`} role="status">
          <strong>{notice.name}: {notice.added === 0 ? "nothing new, these payments were already in your history." : `${notice.added} new ${notice.added === 1 ? "payment" : "payments"} added.`}{notice.credits > 0 && ` ${notice.credits} money-in ${notice.credits === 1 ? "entry" : "entries"} saved under Credit.`}</strong>{" "}
          {notice.spike && "Your recent spending is well above your usual pattern. We're treating it as a one-off, so it's marked unusual and your normal baseline stays as it was. If it continues, your model adapts on its own."}
          {notice.changed && "Your spending pattern has changed and stayed changed, so your model was retrained on the new pattern."}
          <button className="quiet" onClick={() => setNotice(null)} aria-label="Dismiss">Dismiss</button>
        </div>
      )}

      {data && !data.empty && (
        <>
          <KindPager
            deps={[data, txns, filtered.length, shown, shop]}
            debitNote={`${rupees(data.total_spending)} spent`}
            creditNote={data.credits?.has_credits ? `${rupees(data.credits.total)} received` : "nothing found yet"}
            debit={
              <>
                <h2 className="kind-title">Debit: money you spent</h2>
          <section className="story">
              <h1>
                {rupees(data.total_spending)} <span>across {data.profile.total_transactions} payments</span>
              </h1>
              <p className="muted">
                {day(data.period_analysis.start)} to {day(data.period_analysis.end)} · {data.period_analysis.span_days} days
              </p>
              <dl className="facts">
                <div className="tile t-anom"><dt>Worth a look (Anomaly)</dt><dd className={data.anomalies.length ? "warn" : ""}>{data.anomalies.length}</dd></div>
                <div className="tile t-sub"><dt>Subscriptions</dt><dd>{data.subscriptions.length}</dd></div>
                <div className="tile t-cost"><dt>Subscription cost / month</dt><dd>{rupees(data.subscription_summary.monthly_cost)}</dd></div>
                {data.spending_by_merchant[0] && (
                  <div className="tile t-shop"><dt>Top shop</dt><dd className="small">{data.spending_by_merchant[0].merchant}<span>{rupees(data.spending_by_merchant[0].total)}</span></dd></div>
                )}
              </dl>
              <Attention summary={data.anomaly_summary} />
              <h2 className="sub-h">What else we noticed</h2>
              <ul className="insights">
                {data.insights.map((t, i) => <li key={i}>{t}</li>)}
              </ul>
            </section>
  
            <section>
              <h2>Spending by {data.spending_trend.granularity}</h2>
              <div className="chart">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={data.spending_trend.points} margin={{ left: 0, right: 8, top: 8 }}>
                    <CartesianGrid vertical={false} stroke="var(--rule)" />
                    <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 12 }} interval="preserveStartEnd" />
                    <YAxis tickLine={false} axisLine={false} width={56} tick={{ fill: "var(--muted)", fontSize: 12 }}
                      tickFormatter={(v) => (v >= 1000 ? `₹${v / 1000}k` : `₹${v}`)} />
                    <Tooltip cursor={{ fill: "var(--wash)" }} formatter={(v) => [rupees(v), "Spent"]}
                      contentStyle={{ background: "var(--paper)", border: "1px solid var(--rule)", borderRadius: 6 }} />
                    <Bar dataKey="amount" fill="var(--teal)" radius={[3, 3, 0, 0]} maxBarSize={44} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
  
            <Periods periods={data.periods} granularity={data.spending_trend.granularity} />
  
            <Pager
              anomalies={data.anomalies}
              shops={data.spending_by_merchant}
              categories={categories}
              subs={data.subscriptions}
              subSummary={data.subscription_summary}
              activeShop={shop}
              onPickShop={(m) => { setShop(m); setShown(PAGE); document.getElementById("payments")?.scrollIntoView({ behavior: "smooth" }); }}
            />
  
            <section id="payments">
              <h2>All payments</h2>
              <div className="filters">
                <input type="search" placeholder="Search shop or category" value={query}
                  onChange={(e) => { setQuery(e.target.value); setShown(PAGE); }} aria-label="Search payments" />
                {shop && <button className="chip" onClick={() => setShop(null)}>{shop} ✕</button>}
                <span className="muted">{filtered.length} payments, {rupees(filteredTotal)}</span>
              </div>
              <div className="scroll">
                <table>
                  <thead><tr><th>Date</th><th>Shop</th><th>Category</th><th className="num">Amount</th></tr></thead>
                  <tbody>
                    {filtered.slice(0, shown).map((t) => (
                      <tr key={t.transaction_id}>
                        <td>{day(t.timestamp)}</td><td>{t.merchant}</td><td>{t.category}</td>
                        <td className="num">{rupees(t.amount)}</td>
                      </tr>
                    ))}
                    {filtered.length === 0 && <tr><td colSpan="4" className="muted">No payments match.</td></tr>}
                  </tbody>
                </table>
              </div>
              {shown < filtered.length && (
                <button className="quiet" onClick={() => setShown(shown + PAGE)}>Show {Math.min(PAGE, filtered.length - shown)} more</button>
              )}
            </section>
              </>
            }
            credit={<CreditView credit={data.credits || { has_credits: false }} />}
          />

          <section>
            <UploadForm onUploaded={uploaded} compact />
          </section>

          <details className="health">
            <summary>How your model is doing</summary>
            <p>
              {data.model_status.has_personal_model
                ? `Your own model has learned from ${data.model_status.trained_on_count} of your payments (version ${data.model_status.latest_version}). It uses nobody else's data.`
                : "You don't have enough payments yet for your own model, so a general starter model is flagging things for now. Your own model is built automatically after 20 payments."}
            </p>
            <p>
              Spending pattern: {data.drift_status.drift_detected ? "changed recently" : "steady"}
              {data.drift_status.last_checked ? `, last checked ${day(data.drift_status.last_checked)}` : ""}.
            </p>
            <button className="quiet" onClick={retrainNow}>Rebuild my model now</button>
          </details>
        </>
      )}
    </div>
  );
}
