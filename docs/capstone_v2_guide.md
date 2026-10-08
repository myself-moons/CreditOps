# CreditOps v2 — The Complete Beginner-Friendly Guide
## Continuous Drift Observatory & Governed Retraining Platform

Welcome to the complete guide to **CreditOps v2**! 

Whether you are a business stakeholder, a risk officer, a manager, or someone who just learned the word "Drift" yesterday, this document was written for you. It explains **why** this project exists, **what** every number and chart on your screen means, and **how** to use and run it yourself.

---

## Table of Contents
1. [The "Explain It Like I'm Five" Intro: What is Drift & Why Does It Matter?](#1-the-explain-it-like-im-five-intro-what-is-drift--why-does-it-matter)
2. [CreditOps v1 vs CreditOps v2: What Changed?](#2-creditops-v1-vs-creditops-v2-what-changed)
3. [How Does CreditOps v2 Run? (Do I Need to Run Files or Does It Run Automatically?)](#3-how-does-creditops-v2-run-do-i-need-to-run-files-or-does-it-run-automatically)
4. [Complete Page-by-Page & Element-by-Element UI Tour](#4-complete-page-by-page--element-by-element-ui-tour)
   - [Page 1: System Overview (Landing Page)](#page-1-system-overview-landing-page)
   - [Page 2: Live Inference (Transaction Scoring)](#page-2-live-inference-transaction-scoring)
   - [Page 3: Dataset Profile (Data & Stationarity)](#page-3-dataset-profile-data--stationarity)
   - [Page 4: Pipelines & Runs (Offline MLflow Experiments)](#page-4-pipelines--runs-offline-mlflow-experiments)
   - [Page 5: Transaction Logs (Live Audit Trail)](#page-5-transaction-logs-live-audit-trail)
   - [Page 6: The Capstone Observatory (The Core of v2)](#page-6-the-capstone-observatory-the-core-of-v2)
     - [Hero Banner & Status](#hero-banner--status)
     - [Tab 1: Overview & Acceptance Criteria](#tab-1-overview--acceptance-criteria)
     - [Tab 2: Scenario Explorer](#tab-2-scenario-explorer)
     - [Tab 3: Policy Comparison](#tab-3-policy-comparison)
     - [Tab 4: Governance & Audit Trail](#tab-4-governance--audit-trail)
     - [Tab 5: Live Predictions & Stream Simulation](#tab-5-live-predictions--stream-simulation)
5. [The Big Picture: How to Interpret the Project Results](#5-the-big-picture-how-to-interpret-the-project-results)
6. [Step-by-Step Guide: How to Use CreditOps v2 on Your Computer](#6-step-by-step-guide-how-to-use-creditops-v2-on-your-computer)

---

## 1. The "Explain It Like I'm Five" Intro: What is Drift & Why Does It Matter?

Imagine you run a store, and you hire a security guard to spot shoplifters.
You show the guard pictures of known thieves who always wear oversized trench coats in the winter.
For months, the guard is brilliant at catching them.

### Then Spring Comes...
- **The customers change:** People start wearing sunglasses and light t-shirts. That is **Covariate Shift** (the input data looks different, but people are still normal customers).
- **The shoplifters change their tactics:** Thieves realize trench coats get them caught, so they start dressing in business suits or targeting cosmetics instead of jewelry. That is **Concept Drift** or a **Novel Pattern**.
- **A holiday rush hits:** Suddenly there are 3 times more people in the store. That is a **Fraud Rate Shift**.

If the security guard refuses to learn anything new, the store loses thousands of dollars to thieves (**Never Retrain** = failure).

### But Retraining Blindly is Dangerous Too!
What if the security guard panics and decides: *"Anyone carrying an ice cream cone is now a thief!"*
Suddenly, hundreds of innocent paying customers get kicked out of the store, causing chaos and losing money (**Naive Retraining** = false positive disaster).

**CreditOps v2 is the wise supervisor** who watches the security guard, spots when the store environment changes, tests new guards in a back room before putting them on the floor, and immediately steps in if a new guard starts making silly mistakes.

---

## 2. CreditOps v1 vs CreditOps v2: What Changed?

| Feature | CreditOps v1 (The Foundation) | CreditOps v2 (The Capstone Observatory) |
| :--- | :--- | :--- |
| **What was it?** | A static fraud classifier trained once on historical data with an interactive scoring API. | A continuous, real-time MLOps testing ground and governance platform. |
| **How did it detect drift?** | An offline monthly batch script that measured static PSI numbers. | Real-time multi-signal detectors (covariate shift, novelty detectors, delayed PR-AUC drops). |
| **What happened when drift occurred?** | It just showed a warning; a human had to decide what to do manually. | An automated economic brain calculates if retraining is worth the money ($200 cost vs expected loss avoided). |
| **How were new models deployed?** | Automatically replace the old model with whatever had the best training score. | **Governed Validation Gate**: Tests challenger on a secret buffer, recalibrates thresholds, shadow monitors in real-time, and can **roll back** in seconds if costs spike! |
| **Data Storage** | A single flat text file (`predictions.jsonl`). | A dual-engine `LogStore` (Google Firebase Realtime Database + SQLite) with cryptographic audit logs. |

---

## 3. How Does CreditOps v2 Run? (Do I Need to Run Files or Does It Run Automatically?)

This is one of the most common questions: **Where does the magic happen?**

### A. The GitHub Actions Workflow (`.github/workflows/ci.yml`)
- **What it is:** This is the automated robot in GitHub's cloud.
- **When does it run?** Every time code is pushed or a Pull Request is opened.
- **What does it do?** It runs a security audit (`pip-audit`) to check for vulnerable dependencies, and runs all 119 automated tests (`pytest`) in a strictly isolated local SQLite database.
- **Does it run the live banking simulation?** **No.** Running hundreds of thousands of streaming transactions takes substantial time, so GitHub Actions only checks that all software components pass tests and remain secure.

### B. The Live Application (FastAPI & Render)
- **Local:** Runs via `uvicorn src.main:app --host 127.0.0.1 --port 8000`.
- **Cloud:** Deployed live on Render at [https://creditops.onrender.com](https://creditops.onrender.com).
- Whenever you open `http://127.0.0.1:8000/observatory`, the web browser displays the Observatory console.

### C. The Demo Populator (`scripts/seed_demo.py`)
- **Do you need to run this?** **Yes, once** when you want to see the rich evaluation data in the dashboard!
- Running `python scripts/seed_demo.py` takes the authentic results generated during the frozen evaluation and populates the database (Firebase or SQLite) so all charts and metrics are immediately viewable.

### D. The Full Experiment Runner (`experiments/run_all.py`)
- **What is it?** The heavy-duty research engine.
- **What does it do?** It runs all 48 benchmark runs (4 drift scenarios $\times$ 6 retraining policies $\times$ 2 random seeds) across 89 two-day streaming windows (totaling ~350,000 transactions each).
- It has resume support: if stopped midway, it picks right back up where it left off.

---

## 4. Complete Page-by-Page & Element-by-Element UI Tour

CreditOps features 6 unified pages, all connected through a sleek dark-mode sidebar.

---

### Page 1: System Overview (Landing Page)
- **URL:** `/`
- **Purpose:** The welcoming front door. Gives an executive summary of the architecture and quick navigation.
- **What you see:**
  - **Hero Title & Subtitle:** Introduces the project and synthetic data disclosure.
  - **Quick Stats:** Total records (1.85 Million), Baseline Fraud Rate (0.5%), Champion Model (XGBoost).
  - **Launch Buttons:** Direct one-click access to Live Inference, Pipeline Runs, and the Capstone Observatory.

---

### Page 2: Live Inference (Transaction Scoring)
- **URL:** `/predict`
- **Purpose:** Test the fraud detection model with any custom transaction in real time!
- **Input Fields:**
  - *Transaction Date & Time:* e.g., `2020-06-15 14:32:00`.
  - *Amount ($):* e.g., `$850.00`.
  - *Merchant Category:* Grocery, shopping, travel, gas, etc.
  - *Cardholder & Merchant Coordinates:* Latitude/Longitude (the system automatically calculates distance in kilometers using the Haversine formula).
- **The Output Card:**
  - **Fraud Probability Gauge:** A percentage (e.g., `92.4%`).
  - **Decision Tag:** `FRAUD DETECTED` (Red) or `LEGITIMATE` (Green).
  - **Latency:** Inference response time in milliseconds (typically 5–15ms).
  - **Decision Threshold:** Tells you the exact cutoff (e.g., `0.97`) required to flag fraud.

---

### Page 3: Dataset Profile (Data & Stationarity)
- **URL:** `/dataset`
- **Purpose:** Transparency and governance for data scientists and auditors.
- **Key Elements:**
  - **Synthetic Data Disclosure Banner:** Reminds users that all 1,852,394 records were created using the Sparkov synthetic generator (CC0 Public Domain). No real humans or credit card numbers are involved!
  - **Temporal Partitioning Cards:**
    - *Training Split (70%):* 1.29M rows (Jan 2019 to Jun 2020).
    - *Validation Split (15%):* 277k rows (Jun 2020 to Oct 2020).
    - *Test Split (15%):* 277k rows (Oct 2020 to Dec 2020).
  - **Stationarity Metric Card:** Displays the Population Stability Index (PSI = `0.0173`). This mathematically proves that without injected drift, the raw dataset is stable.

---

### Page 4: Pipelines & Runs (Offline MLflow Experiments)
- **URL:** `/dashboard`
- **Purpose:** Compares the offline candidate models trained during development.
- **Key Elements:**
  - **Candidate Model Comparison Table:**
    - *XGBoost (The Champion):* PR-AUC = `0.8722`, ROC-AUC = `0.9977`, Recall = `79.33%`.
    - *Random Forest (Ensemble Benchmark):* PR-AUC = `0.8341`.
    - *Logistic Regression (Linear Baseline):* PR-AUC = `0.3485`.
  - **Why PR-AUC instead of Accuracy?** An explanatory callout reminding you that when fraud is only 0.5%, a dummy model predicting "safe" 100% of the time gets 99.5% accuracy but catches zero fraud! PR-AUC measures true fraud-catching quality.
  - **Observatory Launch Banner:** A vibrant purple banner inviting you into the Capstone Extension v2.

---

### Page 5: Transaction Logs (Live Audit Trail)
- **URL:** `/logs`
- **Purpose:** Telemetry stream of all inference decisions logged to `predictions.jsonl`.
- **Table Columns:**
  - *Timestamp:* Exact moment scored.
  - *Transaction Amount ($):* Value of the transaction.
  - *Predicted Score:* Model probability.
  - *Classification:* `FRAUD` or `LEGIT`.
  - *Latency:* Response time.
  - *Privacy Guarantee:** Shows that zero PII (names, card numbers, addresses) is ever recorded.

---

### Page 6: The Capstone Observatory (The Core of v2)
- **URL:** `/observatory`
- **Purpose:** The crown jewel of Capstone v2. This is the command center where you observe drift, evaluate policies, and manage model safety.

#### Hero Banner & Status
- **Badge:** `CAPSTONE EXTENSION V2 — Evaluation Freeze: eval-v1`. Indicates all evaluation settings were frozen *before* testing to prevent data snooping.
- **Status Pill:** `System Nominal` (Green pulse) indicating detectors and database are functioning smoothly.

---

#### Tab 1: Overview & Acceptance Criteria
This tab gives the high-level summary of the entire experimental benchmark.

1. **Top 4 Metric Cards:**
   - **Evaluation Runs (`48`):** Total benchmark simulations (4 scenarios $\times$ 6 policies $\times$ 2 seeds).
   - **Avg Run Latency (`2.4s`):** How fast the system simulates an entire 89-window (6-month) transaction stream.
   - **Stream Throughput (`36.8 wins/sec`):** The number of 2-day transaction windows processed every second.
   - **Detector Throughput (`14,484 rows/sec`):** High-speed mathematical calculation rate of drift algorithms.

2. **Acceptance Protocol Verification Table:**
   - **AC-1 (Control Safety):** *Target: 0 false retrains.* **Result: 0.0 retrains.** (Passed ✅ — The system never panicked when nothing was wrong).
   - **AC-2 (Governance Protection):** *Target: Never lose more than retrain cost.* **Result: -$200 vs -$370k naive loss.** (Passed ✅ — The gate blocked catastrophic models).
   - **AC-3 (Holdout Disjointness & Delay):** *Target: Zero temporal data leakage with 2-window label delay.* (Passed ✅).
   - **AC-4 (Audit Trail Completeness):** *Target: 100% persisted decisions.* (Passed ✅).
   - **AC-5 (Bitwise Reproducibility):** *Target: Byte-identical results on re-runs.* (Passed ✅).

3. **Consolidated Results Table:**
   - Shows the exact numbers across all scenarios: Retrains, Delays, False Alarms, Costs, and Net Financial Benefits.

---

#### Tab 2: Scenario Explorer
This tab lets you travel through time across any scenario and policy!

- **Controls:**
  - *Scenario Dropdown:* Choose between `control`, `covariate_shift`, `fraud_rate_shift`, or `novel_pattern`.
  - *Policy Dropdown:* Choose from the 6 retraining policies.
- **The Interactive Window Timeline Chart:**
  - **X-Axis:** Windows 0 through 88 (spanning October 2020 through December 2020; 2 days per window).
  - **Dotted Red Line (Window 30):** The exact moment drift was injected!
  - **Lines:**
    - *Blue Line (PR-AUC):* Watch the model's accuracy. In `covariate_shift` or `novel_pattern`, watch the blue line plunge when drift hits! Then watch it jump back up after the governed policy retrains the model!
    - *Red Bars (Window Cost $):* Shows the financial losses per window.
- **Detector Alarms Panel:**
  - Shows which sensor sounded the alarm:
    - *PSI Data Drift:* Triggered when amounts or distances shift.
    - *Novelty Outliers:* Triggered when new cardholder behavior appears.
    - *Unseen Category:* Instantly triggered when a brand new merchant type appears!
- **Policy Decision Feed:**
  - Shows the step-by-step reasoning for every window: `SKIP (stable)`, `ALARM (unseen category)`, `RETRAIN TRIGGERED (expected benefit > $200)`, or `COOLDOWN (waiting 5 windows)`.

---

#### Tab 3: Policy Comparison
This tab proves **which policy is best and why**.

- **Comparison Cards:**
  - **Never Retrain:** Cheaper upfront ($0 compute), but devastating long-term (loses up to $163,000 in missed fraud).
  - **Fixed Calendar Schedule (Naive):** Retrains every 20 windows regardless of need. Costs -$113,620 in unnecessary retrain churn and reacts 24 days too late!
  - **Threshold Policy (Naive):** Only looks at one metric (PSI). Misses novel fraud attacks completely.
  - **Adaptive Policy (Naive):** Detects drift quickly, but deploys blindly without a safety gate. Under fraud rate shifts, it loses -$370,582!
  - **Adaptive Policy (Governed) — The Winner 🏆:**
    - **+$156,997.50** Net Benefit on novel patterns.
    - **+$39,785.00** Net Benefit on covariate shift.
    - **$0.00** False Retrain Churn on stable data.
    - Blocks bad models, saving hundreds of thousands of dollars.

---

#### Tab 4: Governance & Audit Trail
This tab is for Compliance Officers, Regulators, and Security Engineers.

- **Model Registry Table:**
  - Lists every model that was ever trained.
  - Columns: `Model Version`, `Base Commit`, `Status` (`promoted`, `rejected`, `rolled_back`), `Validation PR-AUC`, `Threshold`, and `Shadow Cost`.
  - If a model was rejected at the gate, it shows the exact reason (e.g., *Failed holdout recall margin*).
- **Cryptographic Audit Log:**
  - Displays immutable records of every action taken.
  - **Privacy & Security:** Shows the Actor as a hashed fingerprint (e.g., `risk_owner:a3f81c9b`). Raw API keys are never stored!
- **Emergency Manual Rollback Button:**
  - Allows an authorized Risk Owner to manually roll back a model to the previous champion in case of a real-world emergency. Requires entering a valid `risk_owner` API key!

---

#### Tab 5: Live Predictions & Stream Simulation
- **Purpose:** Watch the stream simulate in real time.
- **Controls:** Play, Pause, and Step buttons to walk through transaction windows one by one and observe the anomaly gauges react.

---

## 5. The Big Picture: How to Interpret the Project Results

If you are presenting this project to leadership or an interviewer, here is your 60-second summary:

> *"In real-world credit card fraud, machine learning models face constant changes (drift). If you never retrain, your model goes blind. If you retrain on a simple calendar schedule, you waste money and react weeks too late. If you retrain automatically without safety checks, you risk deploying uncalibrated models that trigger massive false alarms.
>
> In CreditOps v2, we built and verified a **Governed Adaptive MLOps Architecture**. It uses multi-detector signals to spot fraud drift on Day 0, uses an economic cost gate to decide if retraining is profitable, and passes every new model through a strict validation gate with shadow rollback protection.
>
> In our pre-registered 48-run benchmark, this approach delivered **+$157,000 in net fraud savings** during attacks and **completely prevented a $370,000 catastrophic failure** that naive systems suffered."*

---

## 6. Step-by-Step Guide: How to Use CreditOps v2 on Your Computer

Here is everything you need to run, inspect, and experience CreditOps v2 locally.

### Step 1: Open Your Terminal
Open PowerShell or your terminal in the project directory:
```powershell
cd c:\Users\sharda.poojari\CreditOps
```

### Step 2: Seed the Demo Database
Populate your database (Firebase or SQLite) with the authentic evaluation traces:
```powershell
python scripts/seed_demo.py
```
*You will see a green confirmation showing windows seeded with full provenance!*

### Step 3: Start the Web Application
Launch the FastAPI server:
```powershell
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### Step 4: Open the Web Browser
Open your browser and navigate to:
- **System Overview:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **CreditOps Drift Observatory:** [http://127.0.0.1:8000/observatory](http://127.0.0.1:8000/observatory)
- **Live Scoring Form:** [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)
- **API Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 5: Explore the Observatory Tabs
1. Click **📊 Overview & Criteria** to inspect the 5 passed acceptance criteria.
2. Click **🔍 Scenario Explorer**, pick `novel_pattern` and `AdaptivePolicy(governed)`, and see the exact window where the model detected the attack.
3. Click **⚖ Policy Comparison** to see the financial delta between governed and naive retraining.
4. Click **🛡 Governance & Audit** to review the cryptographic audit records and test the role-based rollback interface.

### Step 6: (Optional) Run the Automated Tests
Whenever you want to verify that all systems, detectors, and APIs are 100% operational:
```powershell
pytest tests/ -q
```
*Expected: 119 passed in ~25 seconds!*

---

*Document version: 2.0 (Phase 5 Final Freeze: `eval-v1`)*  
*CreditOps Architecture Team · October 2026*
