"""
Credits: money that came IN (salary, refunds, transfers received).

Kept completely separate from the spending pipeline: the debit side never
sees these rows, so anomaly scoring, subscriptions and shop totals are
unchanged. Credits only feed the "Credit" view (total, refunds, month by
month, and who the money came from).

Each reader returns a DataFrame with: timestamp, amount, source, kind
(kind is "Refund" when the wording says refund / reversal / cashback,
otherwise "Credit").
"""
import io
import re
import pandas as pd

from .analysis_service import _parse_amount, _join_text_fragments

COLS = ["timestamp", "amount", "source", "kind"]
_REFUND = re.compile(r"\b(?:refund|refunded|reversal|reversed|cashback|cash back|chargeback|returned)\b", re.I)
_PREFIX = re.compile(r"^(?:received from|paid by|from|upi[-/ ]|neft[-/ ]|imps[-/ ]|by transfer[-/ ]?)\s*", re.I)


def _clean_source(text):
    s = _PREFIX.sub("", str(text or "").strip())
    return re.sub(r"\s+", " ", s)[:80] or "Unknown"


def _frame(rows):
    df = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["timestamp", "amount"])
        df = df[df["amount"] > 0].reset_index(drop=True)
    return df


def _row(ts, amount, source, narration=None):
    text = f"{source} {narration or ''}"
    return {"timestamp": ts, "amount": float(amount), "source": _clean_source(source),
            "kind": "Refund" if _REFUND.search(text) else "Credit"}


# ---------- PDF ----------
def credits_from_pdf(pdf_bytes):
    import pdfplumber
    from .statement_extractor import GPAY_BLOCK_RE, _extract_text
    text = _extract_text(pdf_bytes)
    rows = []

    for m in GPAY_BLOCK_RE.finditer(text):
        date_str, direction, name, amount_str, time_str = m.groups()
        if direction != "Received from":
            continue
        try:
            rows.append(_row(pd.to_datetime(f"{date_str} {time_str}", errors="coerce"),
                             float(amount_str.replace(",", "")), name))
        except ValueError:
            continue

    lines = text.split("\n")                     # PhonePe: "<date> ... CREDIT ... amount"
    for i, line in enumerate(lines):
        if "CREDIT" not in line.upper() or "DEBIT" in line.upper():
            continue
        date_m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})", line)
        amt_m = re.search(r"(?:₹|Rs\.?)?\s*([\d,]+\.?\d*)\s*$", line.strip())
        if date_m and amt_m and i > 0:
            try:
                rows.append(_row(pd.to_datetime(date_m.group(1), errors="coerce"),
                                 float(amt_m.group(1).replace(",", "")), lines[i - 1]))
            except ValueError:
                continue

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                header = [re.sub(r"\s+", " ", str(c or "")).strip().lower() for c in table[0]]
                find = lambda keys: next((i for i, h in enumerate(header) if any(k in h for k in keys)), None)  # noqa: E731
                d_i, n_i = find(["date"]), find(["narration", "description", "particulars", "details",
                                                 "paid to", "received from", "payee", "merchant"])
                cr_i, amt_i, ty_i = find(["credit", "deposit"]), find(["amount", "amt"]), find(["type"])
                if d_i is None or n_i is None:
                    continue
                for row in table[1:]:
                    def cell(i):
                        return "" if i is None or i >= len(row) or row[i] is None else str(row[i])
                    narration = _join_text_fragments(cell(n_i).split("\n"))
                    ts = pd.to_datetime(_join_text_fragments(cell(d_i).split("\n")), errors="coerce", dayfirst=True)
                    if cr_i is not None and "type" not in header[cr_i] and cell(cr_i).strip():
                        amount = _parse_amount(cell(cr_i).replace("\n", " "))
                    elif ty_i is not None and amt_i is not None and "credit" in cell(ty_i).lower():
                        amount = _parse_amount(cell(amt_i).replace("\n", " "))
                    else:
                        continue
                    if amount and not pd.isna(ts):
                        rows.append(_row(ts, amount, narration))
    return _frame(rows)


# ---------- CSV / Excel / JSON / Word (already normalised by analysis_service) ----------
def credits_from_frame(df):
    """df: columns already lower_snake_case, with timestamp + merchant mapped."""
    if "timestamp" not in df.columns or "merchant" not in df.columns:
        return _frame([])
    rows = []
    kind_col = next((c for c in ("type", "txn_type", "dr/cr", "cr/dr") if c in df.columns), None)
    cr_col = next((c for c in ("credit", "credit_amount", "deposit", "deposit_amount", "cr") if c in df.columns), None)
    for _, r in df.iterrows():
        ts = pd.to_datetime(r["timestamp"], errors="coerce")
        amount = None
        if kind_col is not None and re.search(r"credit|\bcr\b", str(r[kind_col]), re.I) and "amount" in df.columns:
            amount = _parse_amount(r["amount"])
        elif cr_col is not None and str(r[cr_col]).strip() not in ("", "nan", "None"):
            amount = _parse_amount(r[cr_col])
        if amount and not pd.isna(ts) and amount > 0:
            rows.append(_row(ts, amount, r["merchant"]))
    return _frame(rows)
