import { useEffect, useLayoutEffect, useRef, useState } from "react";

/* Top-level swipe between two big views: Debit (money out) and Credit (money in).
   Tabs, arrows, keyboard and finger-swipe all stay in step. */
export default function KindPager({ debit, credit, debitNote, creditNote, deps }) {
  const [active, setActive] = useState(0);
  const track = useRef(null);
  const panels = useRef([]);
  const target = useRef(null);
  const tabs = [
    { id: "debit", label: "Debit", note: debitNote },
    { id: "credit", label: "Credit", note: creditNote },
  ];

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
    setTimeout(() => { target.current = null; }, 700);
    window.scrollTo({ top: Math.min(window.scrollY, (track.current?.offsetTop || 0) - 90), behavior: "smooth" });
  }
  function onKey(e) {
    if (e.key === "ArrowRight") { e.preventDefault(); go(active + 1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); go(active - 1); }
  }

  // the track is only as tall as the view you are looking at
  useLayoutEffect(() => {
    const t = track.current;
    const el = panels.current[active];
    if (!t || !el) return;
    const fit = () => { t.style.height = `${el.offsetHeight}px`; };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [active, deps]);

  useEffect(() => {
    const onResize = () => { const t = track.current; if (t) t.scrollTo({ left: active * t.clientWidth }); };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [active]);

  return (
    <section className="kinds" aria-label="Debit and credit">
      <div className="kind-bar" role="tablist" aria-label="Money out or money in" onKeyDown={onKey}>
        {tabs.map((t, i) => (
          <button key={t.id} role="tab" id={`kind-${t.id}`} aria-selected={active === i} aria-controls={`kindpanel-${t.id}`}
            tabIndex={active === i ? 0 : -1} className={`${t.id} ${active === i ? "active" : ""}`} onClick={() => go(i)}>
            <span className="k-name">{t.label}</span>
            <span className="k-note">{t.note}</span>
          </button>
        ))}
      </div>
      <div className="kinds-track" ref={track} onScroll={onScroll}>
        {[debit, credit].map((node, i) => (
          <div key={tabs[i].id} className="kind-panel" role="tabpanel" id={`kindpanel-${tabs[i].id}`}
            aria-labelledby={`kind-${tabs[i].id}`} ref={(el) => (panels.current[i] = el)} inert={active !== i ? "" : undefined}>
            {node}
          </div>
        ))}
      </div>
      <p className="swipe-hint muted kind-hint">
        {active === 0 ? "Swipe left for Credit: money that came in" : "Swipe right to go back to Debit"}
      </p>
    </section>
  );
}
