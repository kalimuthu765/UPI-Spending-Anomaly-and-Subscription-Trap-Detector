# UPI Spending Pattern Anomaly & Subscription-Trap Detector

End-to-end MLOps project: anomaly detection on UPI transactions +
subscription/recurring-payment detection, with MLflow tracking,
FastAPI serving, a React dashboard, and drift-triggered retraining.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Real datasets merged into calibration stats | ✅ done — `ml/merge_real_datasets.py` |
| 1 | Synthetic dataset generator (real-data calibrated) | ✅ done — 100k txns, 1000 users, 11 categories, 105 merchants |
| 2 | Preprocessing | ✅ done — `ml/preprocessing.py` |
| 3 | Feature engineering (33 features) | ✅ done — `ml/features.py` |
| 4 | Isolation Forest anomaly model + explanations | ✅ done — `ml/anomaly_model.py` (F1 0.165, ROC-AUC 0.724) |
| 5 | Subscription/recurring detector | ✅ done — `ml/subscription_model.py` (precision 0.99, recall 0.53) |
| 6 | Personalized model (global + per-user) | ✅ done — `ml/personalized_model.py` (F1 0.195, ROC-AUC 0.754) |
| 7 | MLflow tracking + model registry | ✅ done — `mlops/pipelines/train_with_mlflow.py` |
| 8 | FastAPI backend — user-upload analysis flow | ✅ done — `backend/main.py` |
| 9 | Login + PostgreSQL-ready persistence | ✅ done — `backend/auth.py`, `backend/database/`, `backend/services/persist_service.py` |
| 10 | React dashboard | ✅ done — `frontend/` (Vite + React + Recharts) |
| 11 | Drift monitoring | ✅ done — `ml/drift_detection.py` (PSI + KS-test) |
| 12 | Retraining pipeline (per-user, drift-aware) | ✅ done — `backend/services/retrain_service.py` |
| 13 | Docker | ✅ done — `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |

## The upload flow (Phase 8)

The core user-facing feature: `POST /upload/history` — a user uploads
their own transaction history as a CSV and gets back, in one response:
- **Anomalies**, each with a plain-English reason ("Amount is 8.0x higher
  than typical", "Unusual transaction hour")
- **Subscriptions detected**, with cadence and next expected payment date
- **Spending breakdown** — both by category ("₹15,480 on Groceries") and
  by specific shop/merchant ("₹7,367 at Zepto specifically"), plus a
  month-by-month total

It's tolerant of messy real-world exports: only `amount`, `merchant`,
and a date column are strictly required — `category`, `user_id`, and
`transaction_id` are optional and filled in automatically. Users with
20+ transactions get a personal Isolation Forest fit just to them;
below that it falls back to the trained global model.

```bash
uvicorn backend.main:app --reload --port 8000
# then open http://localhost:8000/docs for the interactive API,
# or POST a CSV to /upload/history
```

## Real datasets — merged in

`ml/merge_real_datasets.py` reads the three uploaded Kaggle datasets and
extracts real distributions (not a blind row-merge, per Section 3 of the
plan):
- `upi_transactions_2024.csv` (250k rows) → category shares, per-category
  amount distributions, real fraud rate (0.19%), hour/day patterns
- `upi_transaction_insights_dataset.csv` → payment-mode split (QR/Contact/
  Online/AutoPay), fills categories the main dataset lacks
- `Digital_Payments_2025_Dataset.csv` → month-over-month seasonality curve

These calibrate `generate_synthetic_data.py`, so the synthetic data's
category mix and amounts track the real datasets closely (validated in
`data/processed/calibration_stats.json`).

## Direct training on the real dataset — an important finding

`ml/train_on_real_data.py` trains directly on `upi_transactions_2024.csv`
using its real `fraud_flag` label (Isolation Forest unsupervised vs.
Random Forest supervised). Result: **both score ~0.50 ROC-AUC — chance
level.** `fraud_flag` sits at a near-identical ~0.19% rate across every
category, hour, and device type in the dataset, meaning the label
carries no learnable signal (likely randomly assigned when the dataset
was built). This dataset is still valuable the way it's used in
`merge_real_datasets.py` — real amount/category/payment-mode
distributions — just not for supervised fraud training. Also note: it
has no `user_id` or merchant name, so it can't support per-user
personalization (Phase 6) even in principle. The working, validated
model for this project is the personalized Isolation Forest trained on
the (real-calibrated) synthetic data — see Quick Start below.

## React dashboard (Phase 10)

`frontend/` — Vite + React + Recharts, matching the spec's mockup:
summary cards (total spending, transactions, anomalies, active
subscriptions), a spending-by-category bar chart, an anomaly alert list
with plain-English explanations, and an upcoming-renewals subscription
list with next payment dates. Register/login gate on top, upload form
that shows "Model retrained (drift detected)" when Phase 12 kicks in.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to the backend

# in another terminal:
cd ..
uvicorn backend.main:app --reload --port 8000
```

Verified: `npm run build` compiles cleanly with no errors.

## Drift detection + retraining pipeline (Phase 11-12)

Each user's `/upload/history` call now runs the full spec diagram:

```
New Transactions -> Merge into history -> Drift Detection (PSI + KS-test)
    -> Retrain Trigger? -> [yes] Train + version + save   -> Deploy (used immediately)
                         -> [no]  Reuse the saved model    -> (nothing wasted)
```

Retrain triggers on any of: no personal model yet; significant drift
(PSI >= 0.25, or moderate drift confirmed by a KS-test p-value < 0.05)
between the baseline the saved model was trained on and transactions
added since; or 30+ new transactions accumulated since the last train
(a count-based floor, so a model never goes too stale from volume
alone). Deliberately **transaction-history-based only** — no
time-based trigger. A user who hasn't uploaded anything new has
nothing new to learn from, so there's nothing to retrain on regardless
of how many days passed. Each retrain's `drift_report` includes a
`retrain_reason` field stating exactly which trigger fired.

Verified with a simulated pattern shift: a user uploaded 25 student-like
transactions (avg ₹117) — first personal model trained. A follow-up of
5 more transactions in the SAME pattern correctly did **not** retrain
(reused the saved model — nothing to relearn). Then 15 transactions at
employee-level spending (avg ₹2,650) correctly triggered **significant
drift** (PSI 2.78, KS p-value 0.0) and retrained immediately
(`retrain_reason: "drift_detected"`).

Every check is logged — `GET /me/drift-history` returns the full audit
trail (all drift reports + all model versions) for the logged-in user.

## Login + persistence (Phase 9)

Each account gets its own accumulating history and its own personal
model — proven with a real test: a simulated "student" (avg spend ₹123)
and "employee" (avg spend ₹3,430) each uploaded history, then the
IDENTICAL ₹2,500 transaction was submitted for both:
- **Student**: flagged as an anomaly (20x their typical spend)
- **Employee**: NOT flagged (well within their normal range)

Same transaction, different verdict, purely because each login has its
own persisted baseline — Phase 6's personalization made durable across
sessions instead of recomputed from a single upload each time.

```
POST /auth/register    {email, password, display_name} -> access token
POST /auth/login        (OAuth2 form: username=email, password)
POST /upload/history     [Bearer token] — merges new rows into your
                           stored history, re-analyzes the FULL set
GET  /me/dashboard       [Bearer token] — reads your persisted profile,
                           anomalies, and subscriptions, no re-upload needed
POST /upload/quick-analysis — no login, one-off, nothing saved
```

Database: SQLAlchemy ORM (`backend/database/models.py`), defaults to a
local SQLite file (`upi_app.db`, zero setup) for dev/testing. Point it
at real Postgres with **zero code changes** by setting one env var:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/upi_db"
```

## Quick start

```bash
pip install -r requirements.txt

# Phase 1 — calibrate + generate data
python ml/merge_real_datasets.py         # -> data/processed/calibration_stats.json
python ml/generate_synthetic_data.py     # -> data/synthetic/final_transactions.csv
python ml/preprocessing.py               # -> data/processed/transactions_clean.csv
python ml/features.py                    # -> data/processed/transactions_featured.csv

# Phase 4-6 — models
python ml/anomaly_model.py               # global Isolation Forest
python ml/subscription_model.py          # recurring payment detector
python ml/personalized_model.py          # global + per-user personalization

# Phase 7 — MLflow (logs both model versions, registers the winner)
python mlops/pipelines/train_with_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlops/mlflow/mlflow.db   # view dashboard
```

## Project structure
```
UPI-Anomaly-Detector/
├── data/{raw,processed,synthetic}
├── notebooks/               # EDA, preprocessing, model_training
├── ml/                      # preprocessing, features, models, drift, retraining
├── backend/                 # FastAPI app
├── frontend/                # React dashboard
├── mlops/{mlflow,pipelines,monitoring}
├── docker/
└── tests/
```

## Tech stack
Python · Pandas/NumPy · Scikit-learn (Isolation Forest + Random Forest)
· MLflow · FastAPI · PostgreSQL · React + Recharts · Docker · Evidently

## Refined spec (dynamic periods, insights, confidence)

Later requirements sharpened the project further:

- **Dynamic time-period analysis** (`ml/insights_engine.py`) — no longer
  assumes 6 months of data. Detects the actual uploaded span and adapts:
  a 1-week upload gets "highest-spending day", a 7-month upload gets
  "highest-spending month" + a trend-vs-personal-average sentence, a
  year+ upload adds quarterly analysis. Verified with both a 1-week and
  a 7-month test upload — the wording genuinely changes, not templated.
- **Natural-language insights engine** — sentences like "Amazon is your
  highest-spending merchant — ₹12,450 across 7 transactions" and "You
  spent 24% more than your historical monthly average in March",
  computed from real numbers each time, never hardcoded.
- **Subscription confidence tiers** (High/Medium/Low) in
  `ml/subscription_model.py`, based on repeat count + amount consistency
  — no longer a bare yes/no.
- **Reviewable spending summary** — "You have N recurring payments
  totaling ~₹X/month — worth reviewing if any are unused" (non-judgmental
  phrasing per spec).
- **Expanded API**: `GET /me/transactions`, `/me/insights`,
  `/me/model-status`, `/me/drift-status`, `POST /me/retrain` (manual
  trigger, bypasses the drift check).
- **Expanded dashboard**: Personalized Insights panel, Transaction
  History table, Model Status + Drift Status panels, a manual "Retrain
  now" button.

## PDF statement upload (no manual CSV required)

`POST /upload/history` and `/upload/quick-analysis` now also accept a
**PDF transaction statement** directly — you don't have to hand-build a
CSV. `backend/services/statement_extractor.py` recognizes:

- **Google Pay** statements — parses each "Paid to X ... ₹Y" block; skips
  "Received from" entries since incoming money isn't spending
- **PhonePe** statements — parses DEBIT/CREDIT-labeled transaction blocks
- **Generic bank statements** — falls back to table extraction (Date /
  Narration / Debit columns), the common layout for netbanking PDF exports
- **Labelled card/app statement tables** — a table with a Date, an
  amount-ish column, and a Paid-To/Payee-ish column, where pdfplumber keeps
  each cell's wrapped text as embedded `\n` (e.g. a merchant cell literally
  containing `"Reliance\nSmart\nSupermar\nket"`). Only rows whose `Type`
  column says Debit are kept as spending.

Verified with a synthetic Google Pay statement built from Google's real
published layout: correctly extracted 8 debit transactions (₹11,064.25
total) while excluding 2 incoming "Received from" transactions from the
spending total, then ran the full anomaly + insights + subscription
pipeline on the result — same code path as a CSV upload, no regression
(re-tested a CSV upload immediately after, still works).

**Known limitation**: if a single transaction's row happens to straddle a
PDF page break, pdfplumber's table extraction can drop that one row
(it isn't fully present on either page). Everything else on the page
still extracts correctly — this only affects the rare row sitting right
at a page boundary.

If your statement uses a layout the parser doesn't recognize yet, it
returns a clear error rather than silently extracting nothing or
misreading amounts — share a sample and the parser can be extended.

## Excel (.xlsx/.xls) upload, including word-wrapped exports

`POST /upload/history` and `/upload/quick-analysis` now also accept
`.xlsx`/`.xls` files, handled by `pd.read_excel` in
`backend/services/analysis_service.py`. Column names are matched against
a wide alias list (so "Paid To / Received From", "Date & Time", "Payee",
"Txn Date", "Debit Amount", etc. all map onto the internal `merchant` /
`timestamp` / `amount` columns) and the `amount` column is cleaned of
currency symbols, commas and a stray `+` before being used — "₹1,450.00",
"Rs. 820", and "+₹50.00" all parse correctly.

**Row-wrapped exports**: some real bank/UPI exports (e.g. a GPay/PhonePe
PDF-to-Excel conversion) wrap one logical transaction across *several
physical spreadsheet rows*, because the original cell had word-wrap
enabled and lost its row grouping on export — most cells in the
continuation rows come through blank. `_reconstruct_wrapped_rows()` in
`analysis_service.py` detects this (a reliable per-transaction marker
column, usually `status` or `type`/Debit-Credit, is populated on only one
row per transaction) and reassembles the fragments — a merchant name
spread across 4 rows like `"Reliance" / "Smart" / "Supermar" / "ket"`
correctly becomes `"Reliance Smart Supermarket"` before analysis runs.
Files that aren't fragmented this way pass through unchanged — the
reconstruction only activates when it detects a genuine wrapped-row
pattern, so it never touches a normal clean export.

Verified end-to-end against a real user-submitted bank-statement `.xlsx`
with this exact row-fragmentation pattern: all 9 transactions were
correctly reassembled (dates, amounts, merchant names, payment modes) and
the full pipeline (feature engineering → anomaly scoring → insights) ran
without error on the reconstructed data.

## Any format, really: CSV, Excel, JSON, Word, or PDF

The upload endpoints don't care which format a person's bank, UPI app, or
college/office export happens to produce — `parse_and_validate()` in
`backend/services/analysis_service.py` branches purely on file extension,
and every branch converges on the same internal schema (`timestamp`,
`amount`, `merchant`, `category`, `payment_mode`) before anything else
runs. Supported extensions: `.csv`, `.xlsx`/`.xls`, `.json`, `.docx`/`.doc`,
`.pdf`.

- **JSON** — accepts a plain list of transaction objects, or that same
  list nested under a `"transactions"`/`"data"`/`"records"`/`"results"`
  key (the common shapes real export tools produce), via
  `_read_json_transactions()`.
- **Word (.docx)** — pulls the widest table in the document (via
  `python-docx`), treats its first row as the header, and runs it through
  the same column-alias + row-reconstruction path as Excel — so a bank
  statement pasted into a Word table works the same way a spreadsheet
  does. A document with no table (just paragraphs) gets a clear error
  rather than a silent empty result, since free-text transaction lines
  aren't reliably parseable.

Whichever format arrives, the same personalization and retraining
pipeline runs on the result — the format a person uploads has no bearing
on how their model is trained; only their accumulated transaction history
does (see "Per-user personalization" and "Drift-aware retraining" below).
A person uploading from their college-era GPay PDF today and from their
first-salary bank statement a year later both go through this same path,
and the model adapts because the transactions did, not because the file
type changed.

## Docker (Phase 13)

Four services, one command:

```bash
cp .env.example .env        # edit POSTGRES_PASSWORD and SECRET_KEY
docker compose up --build
```

- **postgres** — Postgres 16, the real database (no more SQLite fallback)
- **backend** — FastAPI on :8000
- **frontend** — the React app, built and served by nginx on :5173, which
  reverse-proxies `/api/*` to the backend container — same code path the
  Vite dev server uses locally, zero frontend code changes needed
- **mlflow** — the tracking UI on :5000, reusing the backend image with
  just the start command overridden

Per-user model artifacts and the MLflow store are mounted as host
volumes (`./ml/artifacts/user_models`, `./mlops/mlflow`) so they survive
a rebuild. Postgres data persists in a named Docker volume.

Switching `DATABASE_URL` from SQLite to Postgres required zero backend
code changes — that was the whole point of building the database layer
on SQLAlchemy from the start (see Phase 9).

**Verification note:** no Docker daemon was available in the environment
these files were built in, so `docker build`/`docker compose up` itself
couldn't be run here. What WAS verified: `docker-compose.yml` parses as
valid YAML with the 4 expected services; every path referenced in both
Dockerfiles (`requirements.txt`, `ml/`, `backend/`, `data/processed/`,
`data/synthetic/`, `frontend/package.json`, `nginx.conf`, etc.) was
checked against the actual repo and exists; `requirements.txt` installs
cleanly with no dependency conflicts in a fresh venv (the same command
the Dockerfile runs); and `backend.main` imports and initializes
correctly in that same fresh environment (the same import Docker's CMD
triggers). Run `docker compose up --build` on your machine for the
final confirmation — if something's off, it'll most likely be small
(e.g. a base image version mismatch) and easy to fix from the build log.

## Experiments tried (and honestly reported, including the ones that didn't help)

Independent notebook research (EDA + 5 anomaly models across all 3 real
datasets, in `mlops/UPI-ML-Experiment/notebooks/` — see git history)
surfaced two techniques worth testing against this project's labeled
data. Both were tested properly rather than assumed to help:

- **Cyclic hour encoding** (`sin`/`cos` of hour instead of raw 0-23) —
  theoretically sound, but measured ROC-AUC *dropped* slightly on both
  the global (0.724→0.707) and personalized (0.754→0.727) models. Not
  used in `FEATURE_COLS`; the columns are still computed in
  `features.py` if anyone wants to experiment further.
- **Ensemble voting** (Isolation Forest + LOF + HBOS, `ml/test_ensemble.py`)
  — best ensemble F1 (0.1448) came in below solo Isolation Forest
  (0.1653) at every vote threshold tested. LOF and HBOS individually
  underperform Isolation Forest here, so combining them dilutes rather
  than strengthens the signal.

Both are genuine negative results, not bugs — documented rather than
silently dropped, because "we tested this and it didn't help, here's
why" is itself a real MLOps/experimentation finding worth reporting.
Isolation Forest alone remains what's actually used in production.

Separately, the notebooks' Dataset 1 experiments (testing `fraud_flag`
against 5 different models) independently confirm this project's
earlier finding that the label carries no learnable signal — precision
stayed ~0.002-0.01 across every algorithm tried, not just Isolation
Forest, which rules out model choice as the explanation.

## Independent exploratory work (mlops/UPI-ML-Experiment/)

`mlops/UPI-ML-Experiment/` holds a second, independently-run exploration
of the same 3 real datasets, kept as supplementary documentation rather
than merged into the production pipeline:

- `notebooks/dataset1.ipynb`, `dataset2.ipynb`, `dataset3.ipynb` — EDA +
  5-model comparison (Isolation Forest, LOF, One-Class SVM, Z-score,
  HBOS) per dataset, plus a `notebooks/final_comparison.ipynb` tying
  the three together
- `models/model_info.json` — formal export run: dataset1 (250k rows,
  22 features) flagged 278 ensemble anomalies (0.11%); dataset2 (620
  rows) flagged 28 (4.52%); dataset3 (500 rows) flagged 18 (3.6%).
  Datasets 2-3 have no ground-truth label, so those rates are
  descriptive/exploratory, not validated precision — unlike dataset1,
  which does have `fraud_flag` and, as above, shows no learnable signal
  regardless of the anomaly rate an unsupervised model reports on it
- `app/predict.py` — a standalone ensemble-voting predictor (≥2-of-4
  models agree) built and tested against dataset3

Not merged into `ml/`: the trained model binaries and the full
250k-row scored CSV from this exploration (~45MB combined) — kept out
to keep this delivery lean, since the production models in `ml/artifacts/`
already are what `backend/` actually loads and serves.


## One-off spikes vs. real lifestyle changes

A person might spend far more than usual for one month and then go back to
normal, or might change for good (college → first job). These need opposite
treatment, so the retraining logic tells them apart:

| What happened | What the system does |
|---|---|
| One heavy month, then normal again | Model is **not** retrained on the spike. The spike transactions are flagged as unusual, the dashboard says it is being treated as a one-off, and the baseline stays unchanged. |
| Spending changed and stayed changed | Detected only when **both halves** of the new transactions are shifted the same way (and at least 1.5x off the usual level, KS-test confirmed). The model then retrains on the new regime only. |
| Stable, with enough new transactions | Routine refresh, with one-off outliers filtered out of the training set. |
| Nothing drifted | The saved model is reused (no wasted retraining). |

Supporting changes: training uses the most recent 500 transactions; outliers
are dropped from the training set (median/MAD on log amounts); the drift
baseline is the exact data the saved model trained on (a small `.npy` file
beside the model, no DB change); PSI uses fewer bins on small samples and
needs the KS test to agree, because PSI alone is mostly noise below ~50
points; and a robust per-user "unusually high amount" rule runs next to the
Isolation Forest, because Isolation Forest cannot flag amounts far beyond
anything in its training range (found in testing: a ₹2,500 spike against a
₹20-150 history was scored as normal). Only unusually **high** spending is
flagged; a tiny purchase is not an alert. Still time-independent: decisions
use transaction history only.

Limit: a spike so large that it is more than half of a person's history
looks like a lifestyle change and is treated as one.

## Dashboard (frontend)

Redesigned around plain sentences and a per-shop ledger rather than a grid of
KPI cards:

- **Story at the top**: total spent, date range, and a few computed sentences
  (highest month, where most money went, top category, repeating payments).
  A month only counts as "full" if the data covers it, so a statement ending on
  the 14th never produces "you spent 96% less".
- **Spending over time** at a granularity that fits the history (days, weeks or
  months); a month still in progress is labelled "(so far)".
- **Where it went, by shop**: every shop with total, number of payments,
  typical size and share of spending. Click a shop to filter the payments table
  to it. Categories are shown the same way.
- **Swipeable views** (swipe on a phone, use the tabs, the arrow buttons or the ← → keys on a computer): *Worth a look* (every unusual payment) → *Spending by shop* (what you normally pay at each shop, plus categories) → *Subscriptions* (what each costs and what has gone out so far). Subscriptions stack into readable rows on narrow screens.
- **Worth a look**: unusual payments with a plain reason, in non-alarming
  wording. A payment that is normal *for that shop* (a Rs 2,000 Amazon order
  when Amazon is usually Rs 700-2,200) is not flagged.
- **Subscriptions**: per-service amount, times paid, total paid so far, next
  expected date and a High/Medium/Low match, plus monthly and yearly cost.
- **All payments** with search and shop filter; upload by drag-and-drop or file
  picker (PDF, Excel, CSV, JSON, Word); light and dark themes; works on phones.
- **How your model is doing**: plain-language status, collapsed by default.

Run: `cd frontend && npm install && npm run dev` (backend on port 8000).
Fonts load from Google Fonts and fall back to system fonts when offline.

## Anomalies as the headline

The dashboard opens with an **"N payments look unusual for you"** block directly under the total. It states the biggest unusual payment in plain words (e.g. "₹1,00,000 at Apple Store on 5 Sept 2026 — 223× your usual payment of ₹448"), then lists unusual payments **month by month** (count and rupee total, latest month first), and what share of total spending they make up. The "See the payments" button jumps to the full list, which is also grouped by month. Wording is deliberately non-alarmist: these are prompts to check, never "fraud". Data comes from `anomaly_summary` and `anomalies` in the dashboard payload.

## Subscriptions of any period

Detection works from real charge dates, so a subscription can renew weekly, fortnightly, monthly, every 10 weeks, quarterly, every 6 months or yearly. It needs a steady gap and a steady amount (3+ charges, or 2 for cycles of ~3 months or longer with an identical amount, shown as "Low"). Same-amount near-daily buys (e.g. a tea stall) are treated as habits, not subscriptions. Monthly/yearly cost is normalised to the real period. A yearly plan seen only once in the upload can't be recognised yet.

## Month-by-month view

`periods` in the dashboard payload lists each month (or week / day for short uploads), newest first: total, number of payments, top 3 shops, and how much of it was unusual.

## Categories when the statement has none

Google Pay / PhonePe / bank PDFs usually carry only the payee name. `ml/categorizer.py` guesses a category from the shop name (Food, Groceries, Fuel, Travel, Bills, Entertainment, Shopping, Health, Education) with ordered keyword rules; anything unmatched stays "Other". A category given by the statement is never overwritten. Older uploads are re-categorised when the dashboard loads. Everyday habits (petrol, groceries, meals, rides) are not treated as subscriptions; shops whose name says "plan" (Netflix, Jio recharge, broadband, Google One ...) are accepted with a looser rhythm and never get High confidence on thin evidence.


## Debit and Credit views
The dashboard has two swipeable views. **Debit** is everything about spending (shops, anomalies, subscriptions).
**Credit** is money received: total, refunds, month by month, and a searchable list of who it came from.
Credits are stored in their own `credits` table (created automatically) and never enter spending, anomaly or
subscription analysis. Uploading the same statement twice no longer doubles your spending. Category guesses now
include **Person** (names that look like private people) and more medical words (medicals, chemist, clinic...);
both are guesses from the name only.
