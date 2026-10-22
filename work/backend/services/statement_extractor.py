"""
Statement Extractor — turns a real-world PDF transaction statement
(Google Pay, PhonePe, or a generic bank statement) into the internal
transaction schema, so users don't have to hand-build a CSV.

Supported formats:
  - Google Pay "Transaction statement" PDF — each transaction is a
    block: "<date>  Paid to/Received from <name>  <amount>" followed by
    a time line and a "Paid to/by <bank>" line.
  - PhonePe statement PDF — similar block structure, slightly different
    wording ("Paid to" / "Received from" / "DEBIT" / "CREDIT").
  - Generic bank statement — falls back to pdfplumber table extraction
    (date / narration / debit / credit columns), the common layout for
    netbanking-exported PDF statements.

Only DEBIT (money going out) transactions are kept for spending
analysis — a "Received from" credit isn't spending, and mixing the two
would badly skew totals and the anomaly model's baseline.
"""

import re
import io
import uuid
import pdfplumber
import pandas as pd

from .analysis_service import _parse_amount, _join_text_fragments

# "03 Jan, 2026   Paid to Mamaearth   506.35" then "09:56 AM" on the next line
GPAY_BLOCK_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]{3},?\s+\d{4})\s+"
    r"(Paid to|Received from)\s+(.+?)\s+"
    r"([\d,]+\.?\d*)\s*\n\s*"
    r"(\d{1,2}:\d{2}\s*[AaPp][Mm])",
    re.MULTILINE,
)

# PhonePe tends to print "DEBIT"/"CREDIT" explicitly alongside a date and amount
PHONEPE_BLOCK_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\D{0,30}?"
    r"(DEBIT|CREDIT)\D{0,10}?"
    r"(?:₹|Rs\.?)?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)


def _extract_text(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _try_parse_gpay(text):
    rows = []
    for m in GPAY_BLOCK_RE.finditer(text):
        date_str, direction, name, amount_str, time_str = m.groups()
        if direction != "Paid to":
            continue  # skip incoming money — not spending
        try:
            ts = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
            amount = float(amount_str.replace(",", ""))
        except ValueError:
            continue
        if pd.isna(ts) or amount <= 0:
            continue
        rows.append({
            "transaction_id": str(uuid.uuid4()), "timestamp": ts, "amount": amount,
            "merchant": name.strip(), "category": "Other", "payment_mode": "UPI",
        })
    return rows


def _try_parse_phonepe(text):
    rows = []
    # PhonePe statements print each transaction as a short paragraph; grab the
    # merchant/description line immediately preceding the DEBIT/CREDIT marker
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.search(r"(DEBIT|CREDIT)", line, re.IGNORECASE)
        if not m or "DEBIT" not in line.upper():
            continue
        date_m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})", line)
        amt_m = re.search(r"(?:₹|Rs\.?)?\s*([\d,]+\.?\d*)\s*$", line.strip())
        if not (date_m and amt_m):
            continue
        merchant = lines[i - 1].strip() if i > 0 else "Unknown"
        try:
            ts = pd.to_datetime(date_m.group(1), errors="coerce")
            amount = float(amt_m.group(1).replace(",", ""))
        except ValueError:
            continue
        if pd.isna(ts) or amount <= 0:
            continue
        rows.append({
            "transaction_id": str(uuid.uuid4()), "timestamp": ts, "amount": amount,
            "merchant": merchant, "category": "Other", "payment_mode": "UPI",
        })
    return rows


def _try_parse_generic_table(pdf_bytes):
    """Fallback for bank-statement-style PDFs with an actual table: date /
    narration / debit / credit columns."""
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                header = [str(c or "").strip().lower() for c in table[0]]
                date_idx = next((i for i, h in enumerate(header) if "date" in h), None)
                narr_idx = next((i for i, h in enumerate(header)
                                  if any(k in h for k in ["narration", "description", "particulars", "details"])), None)
                debit_idx = next((i for i, h in enumerate(header) if "debit" in h or "withdrawal" in h), None)
                if date_idx is None or narr_idx is None or debit_idx is None:
                    continue
                for row in table[1:]:
                    try:
                        date_str = row[date_idx]
                        narration = row[narr_idx]
                        debit_str = row[debit_idx]
                        if not debit_str or not str(debit_str).strip():
                            continue
                        ts = pd.to_datetime(date_str, errors="coerce", dayfirst=True)
                        amount = float(str(debit_str).replace(",", "").strip())
                        if pd.isna(ts) or amount <= 0:
                            continue
                        rows.append({
                            "transaction_id": str(uuid.uuid4()), "timestamp": ts, "amount": amount,
                            "merchant": str(narration).strip()[:80], "category": "Other",
                            "payment_mode": "Bank Transfer",
                        })
                    except (ValueError, TypeError, IndexError):
                        continue
    return rows


def _try_parse_labelled_card_table(pdf_bytes):
    """Fallback for 'card statement'-style PDF tables where each cell can
    contain the whole transaction's text word-wrapped with embedded
    newlines (e.g. a 'Paid To / Received From' cell holding
    'Reliance\\nSmart\\nSupermar\\nket', or an 'Amount' cell holding
    '+\\n₹2,000.00' for a credit). One table row = one transaction,
    unlike the row-fragmented spreadsheet export this same layout produces
    when exported as .xlsx — here pdfplumber keeps each logical row intact,
    it's just the individual cells that wrap.

    Column headers are matched loosely (merchant/payee/paid-to-ish,
    amount-ish, date-ish, a debit/credit 'type' column, status) so this
    isn't tied to one bank's exact wording."""
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                header = [re.sub(r"\s+", " ", str(c or "")).strip().lower() for c in table[0]]

                def find_col(keywords):
                    return next((i for i, h in enumerate(header)
                                 if any(k in h for k in keywords)), None)

                date_idx = find_col(["date"])
                amount_idx = find_col(["amount", "amt"])
                merchant_idx = find_col(["paid to", "received from", "payee", "merchant",
                                          "narration", "description", "particulars", "details"])
                type_idx = find_col(["type"])
                mode_idx = find_col(["payment mode", "mode"])

                if date_idx is None or amount_idx is None or merchant_idx is None:
                    continue

                for row in table[1:]:
                    def cell(i):
                        if i is None or i >= len(row) or row[i] is None:
                            return ""
                        return str(row[i])

                    type_val = cell(type_idx).strip().lower()
                    if type_idx is not None and "credit" in type_val:
                        continue  # only debits count as spending

                    date_str = _join_text_fragments(cell(date_idx).split("\n"))
                    merchant = _join_text_fragments(cell(merchant_idx).split("\n"))
                    amount = _parse_amount(cell(amount_idx).replace("\n", " "))
                    mode = _join_text_fragments(cell(mode_idx).split("\n")) if mode_idx is not None else "UPI"

                    ts = pd.to_datetime(date_str, errors="coerce")
                    if pd.isna(ts) or amount is None or pd.isna(amount) or amount <= 0 or not merchant:
                        continue
                    rows.append({
                        "transaction_id": str(uuid.uuid4()), "timestamp": ts, "amount": amount,
                        "merchant": merchant, "category": "Other", "payment_mode": mode or "UPI",
                    })
    return rows


def extract_transactions_from_pdf(pdf_bytes):
    """Returns a DataFrame in the internal transaction schema, or raises
    ValueError with a clear message if no recognizable format matched."""
    text = _extract_text(pdf_bytes)

    rows = _try_parse_gpay(text)
    source = "google_pay"

    if not rows:
        rows = _try_parse_phonepe(text)
        source = "phonepe"

    if not rows:
        rows = _try_parse_generic_table(pdf_bytes)
        source = "generic_bank_table"

    if not rows:
        rows = _try_parse_labelled_card_table(pdf_bytes)
        source = "labelled_card_table"

    if not rows:
        raise ValueError(
            "Could not recognize this PDF's transaction format. Supported: Google Pay "
            "statements, PhonePe statements, and bank statements with a Date/Narration/"
            "Debit table. If your statement uses a different layout, try exporting it as "
            "CSV/Excel instead, or share a sample so the parser can be extended."
        )

    df = pd.DataFrame(rows)
    return df, source
