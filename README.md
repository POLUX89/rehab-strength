# 🏋️‍♂️ Rehab Strength Dashboard

[![Live app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rehab-strength.streamlit.app)
[![CI](https://github.com/POLUX89/rehab-strength/actions/workflows/ci.yml/badge.svg)](https://github.com/POLUX89/rehab-strength/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.53-FF4B4B.svg)](https://streamlit.io)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

A personal analytics dashboard built with **Streamlit** to track **workouts, sleep, and recovery** over time, with a strong focus on **data integrity, transparency, and rehabilitation monitoring**.

**▶️ Live app: [rehab-strength.streamlit.app](https://rehab-strength.streamlit.app)**

**🧪 No data? Click "Try with synthetic data" inside the app** — it ships with a
fully synthetic demo dataset (zero real health data), so every tab, model and
SHAP plot is explorable in one click.

## 🎥 Demo
https://github.com/user-attachments/assets/7041db62-8f85-4449-a515-f1960cbdb90b

---

## 🎯 Why this project exists

This app was created out of a **real personal need**.

After experiencing a **Cerebrovascular Accident (CVA)**, I needed a reliable way to:
- track physical rehabilitation progress,
- monitor training load and recovery,
- understand how sleep and physiological signals affect performance,
- and **avoid misinterpreting incomplete or delayed data**.

Most fitness apps show numbers without context.
This dashboard is designed to show **what the data actually represents, how recent it is, and how confident we should be when interpreting it**.

Moreover, to practice statistical tests and machine learning models to perform a diagnostic analysis where the key question is *why* it happened.
---

## 🧠 What the dashboard does

The app integrates **three independent data sources**:

### 🏋️ Workouts (Strong)
- Exercise-level tracking
- Estimated 1RM (Epley formula)
- Weekly volume and RPE trends
- Pre vs post CVA comparisons

### 😴 Sleep (Sheets export)
- Sleep score
- Sleep stages (REM / Light / Deep)
- Wake count
- Longitudinal trends

### 🧠 Recovery (Sigmoid model)
- Composite recovery score (0–1)
- Components such as HRV, stress, resting heart rate
- Transparent visualization of trends and variability

## 🔗 Time Series Analysis
- Performs a time series analysis
- ACF and PACF plots
- Statistical tests (ADF, KPSS) for stationarity
- An insight based on the TSA on whether to use time series models

### 📉 Stats
- Location estimate
- Variability estimate
- Outliers detection
- Hypothesis testing

## ⚙️ Models
- Use ML to interpret and explain what happened with a regression problem (Sleep Score) and a classification problem (Good or Bad sleep)
- Supervised models for regression and classification
- Unsupervised models and Dimensionality Reduction for plotting

---

## 📸 Dashboard Preview

### Home
![Home](assets/home1.png)
![Home](assets/home3.png)
![Home](assets/home2.png)

### Workouts
![Workouts](assets/workouts_2.png)

### Recovery
![Recovery](assets/correlation.png)

### Sleep
![Sleep](assets/sleep.png)

### Time Series Analysis
![Time Series Analysis](assets/tsa1.png)
![Time Series Analysis](assets/tsa2.png)

### Stats
![Stats](assets/stats1.png)
![Stats](assets/stats2.png)
![Stats](assets/stats3.png)

### Models
![Models](assets/models1.png)
![Models](assets/models2.png)
![Models](assets/models3.png)
![Models](assets/models4.png)
![Models](assets/models5.png)
![Models](assets/models6.png)
![Models](assets/models7.png)

### Classification
![Classification](assets/classification.png)

---

## 📊 Key features

- **Weekly snapshot** (not noisy daily metrics)
- **Data freshness badges**
  Clearly shows how many days old each data source is
- **Integrity-first design**
  The app never assumes “today’s data” if it doesn’t exist
- **Collapsed upload panel** after data is loaded
- **Correlation analysis** using weekly aggregation (appropriate for 3–4 training days/week)
- Dark mode support 🌙

---

## ⚠️ Data integrity philosophy

This dashboard is intentionally conservative.

- If data is delayed, it is shown as delayed
- No forward-filling or artificial alignment
- Weekly aggregation is preferred when frequencies differ
- Visual cues explain *how confident* we should be in the metrics

This is especially important in a **rehabilitation context**, where misinterpreting trends can lead to poor decisions.

---

## 🧩 Architecture overview

The project separates a **private ingestion pipeline** from a **public dashboard**.

```
Google Sheets ─┐
Garmin (xlsx) ─┼─► rehab_strength.ingest ─► data/processed/*.csv ─► Streamlit app
Strong (csv)  ─┘        (local only)          (git-ignored)         (manual upload)
```

Inside the app:

- Upload → normalize → store in `st.session_state`
- UI widgets never re-read raw files
- All downstream logic reads from validated in-memory data
- Deterministic behavior across reruns and tab changes

### Repository layout

```
├── streamlit_app.py            # Dashboard entrypoint (deployed to Streamlit Cloud)
├── app/
│   ├── helpers/                # Pure stats/transforms + plotting helpers
│   └── tabs/                   # One module per tab; models/ split into
│                               #   regression/, classification/, unsupervised
├── src/rehab_strength/
│   ├── config.py               # Paths & settings, all relative to the repo root
│   ├── gsheets.py              # Google auth — credentials resolved from the environment
│   └── ingest/
│       ├── sleep.py            # Sheets + Garmin → sleep & recovery datasets
│       ├── strong.py           # Strong export → clean workouts
│       └── run_all.py          # Full pipeline
├── data/{raw,processed,external}/   # Health data — git-ignored
├── data/synthetic/             # Synthetic demo dataset — versioned (no real data)
├── models/                     # Trained models — git-ignored
├── reports/figures/            # Generated plots — git-ignored
├── assets/                     # Curated screenshots for this README — versioned
├── notebooks/                  # Exploration — outputs stripped, .ipynb git-ignored
└── tests/
```

---

## 🚀 Tech stack

- **Python 3.13** — pinned via `.python-version` for parity with Streamlit Cloud
- **Streamlit** — dashboard
- **Pandas / NumPy** — data wrangling
- **Matplotlib / Seaborn** — visualization
- **scikit-learn / statsmodels / SHAP** — modeling and diagnostics
- **Google Sheets & Drive API** — sleep ingestion
- **pytest / ruff / pre-commit / GitHub Actions** — quality gates

---

## ⚙️ Getting started

```bash
git clone git@github.com:POLUX89/rehab-strength.git
cd rehab-strength
make setup                 # venv + dependencies + pre-commit hooks
source .venv/bin/activate
make app                   # http://localhost:8501
```

The dashboard runs on **uploaded CSVs** — no credentials required. Drop your three
cleaned CSVs into the upload panel and every tab unlocks. Or skip the uploads
entirely: **"🧪 Try with synthetic data"** loads the bundled synthetic dataset
(regenerate it anytime with `make synthetic`).

### Running the ingestion pipeline (owner only)

Requires a Google service account with read access to the Health Metrics sheets:

```bash
cp .env.example .env       # point GOOGLE_APPLICATION_CREDENTIALS at your key
make ingest                # → data/processed/*.csv
```

Place the Garmin exports (`HRV_status.xlsx`, `Sleep_garmin.xlsx`) and the Strong
export (`strong.csv`) in `data/raw/` first. See [`data/README.md`](./data/README.md).

Then upload the three regenerated CSVs to the app — the dashboard never reads the
filesystem, only what you hand it.

```bash
make test          # run the test suite
make lint          # style checks
make check-secrets # scan for leaked credentials
```

### Running the pipeline automatically (macOS)

[`scripts/run_pipeline.sh`](./scripts/run_pipeline.sh) runs the full ingestion, logs to
`~/Library/Logs/rehab_strength_pipeline.log` and fires a macOS notification — including
when it *fails*, so stale data never passes silently.

Wire it to run whenever the Mac wakes up, using
[sleepwatcher](https://formulae.brew.sh/formula/sleepwatcher):

```bash
brew install sleepwatcher
brew services start sleepwatcher

cat > ~/.wakeup <<'SH'
#!/bin/zsh
"$HOME/dev/rehab-strength/scripts/run_pipeline.sh"
SH
chmod +x ~/.wakeup
```

The script resolves the repo root from its own location, so it works from any clone.
Override the notification app with `REHAB_NOTIFY_APP`; without one it falls back to
`osascript`.

---

## 📈 Versioning

All changes are tracked in [`CHANGELOG.md`](./CHANGELOG.md), following semantic versioning principles.

---

## 🔒 Privacy & security

This project is for **personal use**, and it is built so the code can be public
while the data never is.

- **No personal data in the repo.** Every health file lives under `data/`, which is
  git-ignored. CSV, XLSX and PDF are blocked repo-wide by default.
- **No credentials in the repo.** Google service account keys are resolved at runtime
  from environment variables or Streamlit secrets — never from a file in the tree.
  See [`.env.example`](./.env.example) and [`.streamlit/secrets.toml.example`](./.streamlit/secrets.toml.example).
- **The deployed app holds no secrets.** It reads manually uploaded CSVs and never
  contacts Google.
- **Defense in depth.** `pre-commit` runs `gitleaks` and
  `nbstripout` on every commit; CI runs `gitleaks` across the full history.
- **Notebook outputs are stripped**, since rendered cells embed personal health data.

The ingestion code is published; the data it ingests is not.

---

## 📌 Disclaimer

This tool is **not a medical device**.
It is intended for **personal tracking and insight**, not diagnosis or medical advice.

---

## 🙌 Closing note

This project represents a transition from:
> “just tracking numbers”
to
> **understanding recovery, uncertainty, and progress over time**.

It is both a technical project and part of an ongoing rehabilitation journey.
