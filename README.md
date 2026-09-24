# Hilton Take-Home Project 
# Task 1: Model Benchmarking & A/B Testing — Credit Card Churn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thamil-DSci/int1_Thamil/blob/main/Hilton_ML_Project.ipynb)

> **TL;DR** — My XGBoost solution catches the **same share of churners** as a published Kaggle solution (recall 0.953 vs 0.957, no significant difference) while raising **~35% fewer false alarms**. F1 improves from **0.848 → 0.886** (+4.5%, p = 4.5 × 10⁻⁶), and my model wins in **25 of 25** paired cross-validation folds.

| | |
|---|---|
| **My notebook** | [`Hilton_ML_Project.ipynb`](https://github.com/Thamil-DSci/int1_Thamil/blob/main/Hilton_ML_Project.ipynb) |
| **Baseline (Kaggle)** | [Kaushik Majumder — *Credit Card Customer Churn Prediction*](https://www.kaggle.com/code/kaushikmajumder/credit-card-customer-churn-prediction/notebook) |
| **Dataset** | [BankChurners.csv](https://drive.google.com/file/d/10kj_aIX4zpiv2wLFU5KyVUz7yr7V-jYD/view?usp=drive_link) — 10,127 customers, 19 features, 16.1% churn |

## Contents
1. [Problem Selection](#1-problem-selection)
2. [Solution Implementation](#2-solution-implementation)
3. [Baseline for Comparison](#3-baseline-for-comparison)
4. [A/B Testing](#4-ab-testing-setup)
5. [Comparison and Quantitative Results](#5-comparison-and-quantitative-results)
6. [Analysis and Explanation](#6-analysis-and-explanation)

---

## 1. Problem Selection

| | |
|---|---|
| **Problem category** | Supervised binary classification |
| **Target variable** | `Attrition_Flag` — *Attrited Customer* = 1 (churn), *Existing Customer* = 0 |
| **Class balance** | 1,627 churners (16.1%) vs 8,500 existing customers (83.9%) |

**Business problem.** The bank has seen a steep decline in credit card users. Credit cards earn the bank fee income (annual, balance-transfer, cash-advance, late-payment and foreign-transaction fees), so every customer who leaves is lost revenue. The bank wants to **identify customers who are likely to close their card** — and understand why — so it can act before they leave.

**Why recall matters most.** A missed churner (false negative) is a lost customer; a false alarm (false positive) is only a wasted retention offer. The key metric is therefore **recall on churners**, with precision as a secondary concern.

> **Note:** This project was prepared for an interview evaluation. The Kaggle notebook is used purely as a published reference point for an A/B comparison, not as a critique of the author's work.

---

## 2. Solution Implementation

### Data Split
- **Train / test = 70 / 30**, stratified on `Attrition_Flag`, `random_state=7`:
  ```python
  train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
  ```
- **7,088** training rows (1,139 churners) and **3,039** test rows (488 churners). Stratification keeps the 16.1% churn rate identical in both sets.
- Model selection and tuning used **5-fold stratified cross-validation on the training set**; the test set was kept out of all fitting.

### Preprocessing & Pipeline
| Step | Method | Details |
|---|---|---|
| **Missing values** | `KNNImputer(n_neighbors=5)` | "Unknown" in `Education_Level` (15%), `Marital_Status` (7%) and `Income_Category` (11%) is treated as missing and filled from the 5 most similar customers (Euclidean distance). Fitted on training data only. |
| **Categorical encoding** | One-hot (`pd.get_dummies`, `drop_first=True`) | Categories are label-encoded for the imputer, mapped back to labels, then one-hot encoded (5 categorical columns → 16 dummies). |
| **Scaling** | `StandardScaler` | Zero mean, unit variance; fitted on training data only inside a scikit-learn `Pipeline`, so the same transformation is applied to test data without leakage. |
| **Class imbalance** | `scale_pos_weight=10` | Each churner counts ~10× in the XGBoost loss — no rows discarded (under-sampling) or synthesised (SMOTE). |
| **Features** | All 19 kept | Tree models can use features with weak *linear* correlation through non-linear effects and interactions. |

### Modeling Technique
**Selected algorithm:** `XGBClassifier`, tuned with `RandomizedSearchCV` (50 candidates, 5-fold CV, scored on recall), in a pipeline after `StandardScaler`.

```python
XGBClassifier(n_estimators=50, learning_rate=0.2, gamma=5, subsample=0.7, scale_pos_weight=10)
```

**Rationale**
- **Simpler models could not catch enough churners.** Logistic regression reached only **0.43** test recall. Over- or under-sampling raised it to ~0.80, but precision fell to ~0.45. A single decision tree reached 0.75–0.79. The churn signal is non-linear (transaction count, amount and revolving balance interact), which a linear model cannot capture well.
- **XGBoost had the best recall of all seven algorithms.** After tuning it caught **94.3%** of churners with **82.9%** precision and **96.0%** accuracy on the test set. The other ensembles were more precise (~0.93) but missed 12–25% of churners.
- **RandomizedSearchCV over GridSearchCV.** The grid-searched XGBoost gained only 0.8 recall points (~4 more churners) but lost 10.5 precision points (~82 more false alarms), and took far longer to run.

| # | Model | Train Accuracy | Test Accuracy | Train Recall | Test Recall | Train Precision | Test Precision |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Logistic Regression | 0.88 | 0.88 | 0.42 | 0.43 | 0.70 | 0.72 |
| 1 | Logistic Regression on Oversampled data | 0.83 | 0.81 | 0.84 | 0.79 | 0.82 | 0.45 |
| 2 | Logistic Regression-Regularized (Oversampled data) | 0.71 | 0.80 | 0.57 | 0.55 | 0.78 | 0.42 |
| 3 | Logistic Regression on Undersampled data | 0.82 | 0.82 | 0.82 | 0.80 | 0.82 | 0.46 |
| 4 | Decision Tree with GridSearchCV | 0.94 | 0.92 | 0.82 | 0.75 | 0.81 | 0.74 |
| 5 | Decision Tree with RandomizedSearchCV | 0.93 | 0.91 | 0.82 | 0.78 | 0.75 | 0.70 |
| 6 | Bagging Classifier with GridSearchCV | 1.00 | 0.97 | 0.99 | 0.84 | 1.00 | 0.93 |
| 7 | Bagging Classifier with RandomizedSearchCV | 1.00 | 0.97 | 0.99 | 0.84 | 1.00 | 0.93 |
| 8 | Random Forest with GridSearchCV | 1.00 | 0.95 | 0.97 | 0.75 | 1.00 | 0.93 |
| 9 | Random Forest with RandomizedSearchCV | 1.00 | 0.95 | 0.97 | 0.75 | 1.00 | 0.93 |
| 10 | AdaBoost with GridSearchCV | 0.97 | 0.96 | 0.89 | 0.83 | 0.95 | 0.94 |
| 11 | AdaBoost Tree with RandomizedSearchCV | 0.97 | 0.95 | 0.87 | 0.78 | 0.94 | 0.92 |
| 12 | GradientBoost with GridSearchCV | 0.99 | 0.97 | 0.96 | 0.88 | 0.98 | 0.93 |
| 13 | GradientBoost Tree with RandomizedSearchCV | 0.99 | 0.97 | 0.96 | 0.88 | 0.98 | 0.93 |
| 14 | XGBoost with GridSearchCV | 0.96 | 0.93 | 1.00 | 0.95 | 0.78 | 0.72 |
| 15 | **XGBoost with RandomizedSearchCV (selected)** | **0.98** | **0.96** | **1.00** | **0.94** | **0.90** | **0.83** |
---

## 3. Baseline for Comparison

**Source:** [Kaushik Majumder — *Credit Card Customer Churn Prediction* (Kaggle)](https://www.kaggle.com/code/kaushikmajumder/credit-card-customer-churn-prediction/notebook). It uses the same dataset and the same business goal: maximise recall on churners.

### Baseline Overview
| Stage | Kaggle baseline |
|---|---|
| **Data split** | Train / validation / test = **60 / 20 / 20** (6,075 / 2,026 / 2,026), stratified, `random_state=1` |
| **Dropped columns** | `CLIENTNUM`, the two leakage columns `Naive_Bayes_Classifier_*`, and 5 features judged "uncorrelated" with churn: `Customer_Age`, `Dependent_count`, `Months_on_book`, `Credit_Limit`, `Avg_Open_To_Buy` (14 raw features remain) |
| **Missing values** | "Unknown" (and the junk value `abc` in `Income_Category`) kept as its own `Unknown` category — no imputation |
| **Encoding** | One-hot (`pd.get_dummies`, `drop_first=True`) → 27 features |
| **Scaling** | `RobustScaler` (IQR-based) on the 9 numeric columns |
| **Class imbalance** | **Random under-sampling**: non-churners cut from 5,099 to 976, so the final model trains on **1,952 rows** |
| **Model search** | 7 algorithms × 3 training sets (original, SMOTE, under-sampled), 10-fold CV on recall; top 4 tuned with `RandomizedSearchCV` |
| **Selected model** | `GradientBoostingClassifier(n_estimators=700, max_depth=25, min_samples_leaf=15, min_samples_split=2, max_features='sqrt')` |
| **Reported test result** | Accuracy 0.934 · Recall 0.960 · Precision 0.721 · F1 0.823 · ROC-AUC 0.991 |

**Author's rationale.** The under-sampled XGBoost had the highest validation recall (~99%) but was rejected because its accuracy (~73%) was below the 84% of a model that predicts "no churn" for everyone. GBM was chosen as the best balance: validation recall ~97%, accuracy ~94%, precision ~74%, ROC-AUC ~0.99.

### Side-by-side
| | **My solution** | **Kaggle baseline** |
|---|---|---|
| Features | All 19 | 14 (5 dropped by linear correlation) |
| "Unknown" values | KNN-imputed | Kept as a category |
| Scaling | `StandardScaler`, all features | `RobustScaler`, 9 numeric features |
| Imbalance handling | Class weighting — all data kept | Under-sampling — ~80% of non-churners discarded |
| Model | XGBoost, 50 trees | Gradient Boosting, 700 trees of depth 25 |

---

## 4. A/B Testing 

### Hypothesis
- **Null hypothesis ($H_0$):** there is no significant difference in F1 score between my solution and the baseline: $\mu_{proposed} = \mu_{baseline}$.
- **Alternative hypothesis ($H_1$):** my solution achieves a significantly higher F1 score: $\mu_{proposed} > \mu_{baseline}$.
- **Guardrail:** my recall must not be more than **2 percentage points** below the baseline's — the bank cannot accept missing more churners in exchange for fewer false alarms.

### Evaluation Metrics
| Role | Metric | Why |
|---|---|---|
| **Primary** | **F1 score** (churn class, threshold 0.5) | Both solutions already catch ~95% of churners, so recall alone cannot separate them. F1 balances recall (lost customers) and precision (wasted offers). |
| Guardrail | Recall | The business KPI both solutions optimised. |
| Secondary | Precision, PR-AUC, ROC-AUC, Accuracy | PR-AUC is threshold-free and suited to imbalanced data. Accuracy is reported but not relied on — predicting "no churn" for everyone already scores 84%. |

### Data Splitting & Validation
- **Stratified 5-fold cross-validation, repeated 5 times** with a fixed seed — `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)` — on all 10,127 customers → **25 paired folds** (50 model fits in total).
- **Identical splits:** in every fold both models see exactly the same ~8,101 training rows and ~2,026 test rows (~325 churners).
- **No leakage:** each model runs its full pipeline inside every training fold.
  - **Mine:** KNN imputation → one-hot → `StandardScaler` → XGBoost, trained on all ~8,101 rows.
  - **Baseline:** drop 5 columns → one-hot → `RobustScaler` → under-sampling (~2,600 balanced rows) → Gradient Boosting.
- **Why not compare the published scores directly?** The two notebooks used different splits (70/30 vs 60/20/20), so their reported test scores are not comparable.

---

## 5. Comparison and Quantitative Results

### Performance Comparison Table
Mean over 25 paired folds.

| Metric | Baseline (Kaggle GBM) | Proposed (My XGBoost) | Delta ($\Delta$) | % Improvement | 95% CI of $\Delta$ | Proposed better in |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1 (primary)** | 0.848 | **0.886** | **+0.0385** | **+4.5%** | [+0.0243, +0.0526] | **25 / 25** |
| Recall (guardrail) | 0.957 | 0.953 | −0.0045 | −0.5% | [−0.0189, +0.0098] | 6 / 25 |
| Precision | 0.761 | **0.829** | +0.0677 | +8.9% | [+0.0458, +0.0896] | 25 / 25 |
| PR-AUC | 0.943 | **0.965** | +0.0223 | +2.4% | [+0.0102, +0.0345] | 25 / 25 |
| ROC-AUC | 0.989 | **0.993** | +0.0038 | +0.4% | [+0.0018, +0.0057] | 25 / 25 |
| Accuracy | 0.945 | **0.961** | +0.0159 | +1.7% | [+0.0104, +0.0214] | 25 / 25 |

> **In customer terms** (per fold of 2,026 customers, ~325 churners): both models catch about **310 churners**, but mine raises **~64 false alarms vs ~98** for the baseline — **about 35% fewer wasted retention offers**.

### Statistical Significance Test
| | |
|---|---|
| **Test type** | One-sided **paired t-test** on the 25 per-fold differences, with the **Nadeau–Bengio correction** (CV folds share training data, so a plain t-test would overstate significance) |
| **p-value (F1)** | **4.5 × 10⁻⁶** ($\alpha = 0.05$) |
| **Result** | ✅ **Statistically significant** — $H_0$ rejected; my solution has a higher F1 score |
| **Recall guardrail** | ✅ **Passed** — difference −0.45 pts (p = 0.74, not significant); lower 95% CI bound −1.89 pts stays inside the −2 pt margin |
| **Other metrics** | Precision, PR-AUC, ROC-AUC and accuracy are all significantly higher (p < 0.001), better in 25 / 25 folds |

<img width="1589" height="393" alt="A/B test results: difference per metric with 95% CI, and F1 and precision per fold" src="https://github.com/user-attachments/assets/9da9b4fb-4067-48dc-b5d8-499674ac6b61" />

*Left: mean difference per metric with 95% CI (dotted line = −2 pt recall margin). Middle and right: F1 and precision per fold — each grey line joins the same fold.*

---

## 6. Analysis and Explanation

### Insights on Differences
My solution catches **the same churners** as the baseline with **far fewer false alarms**. Its precision–recall performance is higher at every threshold (PR-AUC 0.965 vs 0.943), so this is **better ranking of customers**, not just a different threshold. Three reasons:

1. **Keeping all the data vs under-sampling.** The baseline discards ~70% of the non-churners in each training fold (8,101 → ~2,600 rows). That pushes recall up, but the model sees far fewer loyal customers and over-flags them. My solution handles the imbalance with **class weighting** and learns from every row.
2. **Keeping all features.** The baseline dropped `Customer_Age`, `Dependent_count`, `Months_on_book`, `Credit_Limit` and `Avg_Open_To_Buy` because their *linear* correlation with churn was near zero. Tree models use non-linear effects and interactions, so weak linear correlation is not a good reason to drop a feature. The customers only the baseline wrongly flags tend to be **younger and newer to the bank** — exactly the information it removed.
3. **Model capacity and regularisation.** The baseline's 700 trees of depth 25 on ~2,600 rows tend to memorise the training data. My XGBoost (50 trees, `gamma=5`, `subsample=0.7`) is smaller and regularised, and its precision varies less from fold to fold.

### Strengths vs. Weaknesses
| ✅ Strengths | ⚠️ Weaknesses |
|---|---|
| ~35% fewer wasted retention offers at the same recall | **Recall is not better** — slightly lower on average (−0.45 pts, not significant); the CI lower bound (−1.89 pts) is close to the 2-pt margin |
| Significantly higher F1, precision, PR-AUC and accuracy in **all 25 folds** | **KNN imputation** adds a step at scoring time and treats category codes as numbers — a heuristic for nominal variables |
| Uses all customers and all features — no data discarded | **Mild over-fitting:** train recall 1.00 vs ~0.95 on unseen data |
| Smaller, faster model (50 trees vs 700 deep trees) and more stable across folds | **Tuned on recall only**, with the default 0.5 threshold rather than one chosen from business costs |

### Gaps & Future Improvements
1. **Choose the threshold from business costs** — pick the threshold (inside CV) that minimises *(cost of a lost customer × missed churners) + (cost of an offer × false alarms)*, or that hits a target recall such as ≥ 0.96. Because my model ranks customers better, it can match the baseline's recall while keeping higher precision.
2. **Tune on PR-AUC or F2 instead of raw recall**, so the search accounts for the cost of false alarms.
3. **Simplify categorical handling** — keep "Unknown" as its own category, or use native categorical support (LightGBM / CatBoost).
4. **Feature engineering** — e.g. average transaction value (`Total_Trans_Amt / Total_Trans_Ct`), recent-activity drop (Q4/Q1 change × months inactive).
5. **Nested cross-validation for tuning**, so the reported score is not measured on the data used to choose the model.
6. **Explainability** — SHAP values per customer, so the retention team sees *why* someone is flagged.
7. **Online A/B test** — randomise flagged customers into offer / no-offer groups and measure **customers actually retained**. Offline metrics are only a proxy for business impact.





# Task 2: Flight Data Simulation & Analysis



In this task I built a small Python program in two parts. First it **creates** fake flight data (about 5,000 JSON files). Then it **reads all of that data back, cleans it, and answers a few questions** about it: which cities get the most passengers, how long those flights take, and which cities gain or lose the most people overall.

Everything is in one notebook: [`Hilton_TakeHome_Project_Task2.ipynb`](Hilton_TakeHome_Project_Task2.ipynb). It runs from top to bottom with no extra files needed.

---

## Results at a glance

| What | Result |
|---|---|
| Files created | **5,000** JSON files |
| Total flight records processed | **375,234** |
| Bad ("dirty") records found and removed | **3,618** (0.96%) |
| Clean records used for analysis | **371,616** |
| Total runtime of the analysis phase | **~1,900 ms** (about 2 seconds) |
| Destination with the most passengers arriving | **Nice** (459,520 passengers) |
| City with the **most** passengers remaining | **Abuja** (+68,552) |
| City with the **fewest** passengers remaining | **Bucharest** (−55,946) |
| Self-checks passed | **8 / 8** ✅ |

---

## Phase 1 — Creating the flight data

The goal here was to make realistic-looking test data that the second phase can work on.

**What I did, step by step:**

1. **Picked a list of cities.** I started with 200 city names and randomly chose between 100 and 200 of them for this run (it picked 181).
2. **Gave each city a spot on a map.** Each city gets random coordinates, so a flight's duration depends on how far apart the two cities are, plus a bit of random variation. Without this, every flight would be a random length and the averages would mean nothing.
3. **Made the flight records.** Each record has five fields:

   | Field | Example |
   |---|---|
   | `date` | `2026-09-17` |
   | `origin_city` | `Abu Dhabi` |
   | `destination_city` | `San Diego` |
   | `flight_duration_secs` | `20738` |
   | `#_of_passengers_on_board` | `156` |

4. **Added some "dirty" records on purpose.** About 0.5–1% of records (0.96% in this run) have one to three fields left empty (`null`), to test the cleaning step.
5. **Saved the files.** Each file holds 50–100 flights leaving one city.

**One problem I had to solve with the file names.** The task asks for names like `09-26-London-flights.json` (month-year, then city). But there are only up to 200 cities and one month, so at most 200 files can have different names. If I saved 5,000 files in one folder, most would overwrite each other. To fix this, I put the files into numbered folders (`batch_0001`, `batch_0002`, …). Each folder has one file per city, with the exact name the task asks for:

```
/tmp/flights/
├── batch_0001/
│   ├── 09-26-London-flights.json
│   ├── 09-26-Paris-flights.json
│   └── ...
├── batch_0002/
│   └── ...
```

While generating, the program also keeps a count of exactly how many records and dirty records it wrote. I use these numbers at the end to check that Phase 2 got everything right.

---

## Phase 2 — Cleaning and analyzing the data

A timer starts at the beginning of this phase and stops at the end, so the runtime covers loading, cleaning and all the calculations.

### Step 1: Load all the files
I read every JSON file and put all the flight records into **one pandas table** (`df_raw`). I collect the records in a list first and build the table once at the end, which is much faster than building 5,000 small tables and joining them. I also keep the file name next to each record, so any bad record can be traced back to where it came from.

### Step 2: Find and remove the bad records
A record is "dirty" if **any** of its five fields is empty. I counted them, then removed them.

**Why remove instead of fix?** You can't reliably guess a missing destination, flight time or passenger count. Making up values would quietly skew the averages and the passenger totals, so it's safer to leave those records out.

The clean table is called **`df`**, and everything after this step uses it. I also set proper data types (whole numbers, dates, and a memory-saving type for city names).

### Step 3: Report the counts
- Total records processed: **375,234**
- Dirty records: **3,618** (0.96%)
- Runtime: shown at the end, once all the steps are finished

### Step 4: Top 25 destinations — average and P95 flight time
1. I added up the passengers arriving at each city and kept the **top 25**.
2. For each of those 25 cities I calculated:
   - **AVG**: the average flight duration into that city.
   - **P95**: the 95th percentile, meaning 95% of flights into that city are shorter than this. It shows how long the *longer* flights take, not just the typical one.

The notebook shows these as a table (in seconds, and in hours:minutes) and as a chart.

### Step 5: Passenger balance for each city
Every city starts at **0**. For every flight:
- the **destination gains** the passengers on board;
- the **origin loses** the same number.

So each city's result is simply **passengers arrived − passengers departed**.

- **Most passengers remaining:** Abuja, **+68,552** (more people flew in than out)
- **Fewest passengers remaining:** Bucharest, **−55,946** (more people flew out than in)

**Why can the number be negative?** Because every city starts at 0, the balance is really the *net flow* of people, not a head-count. A negative number just means more people left that city than arrived.

**A built-in sanity check:** every passenger leaves one city and arrives in another, so all the balances together must add up to exactly **0**. They do.

---

## Checking my own work

At the end, the notebook compares Phase 2's results with what Phase 1 actually wrote:

| Check | Result |
|---|---|
| Records processed = records generated (375,234) | ✅ PASS |
| Dirty records found = dirty records generated (3,618) | ✅ PASS |
| 5,000 files were created | ✅ PASS |
| Every file has 50–100 records | ✅ PASS |
| City pool size is between 100 and 200 (181) | ✅ PASS |
| No flight goes from a city to itself | ✅ PASS |
| No empty values are left in the clean data | ✅ PASS |
| All passenger balances add up to 0 | ✅ PASS |

---

## Design choices in short

| Decision | Why |
|---|---|
| Numbered batch folders | Keeps the exact file names from the task without files overwriting each other |
| Distance-based flight times | Makes the AVG and P95 results meaningful instead of random |
| Remove dirty records rather than fill them in | Guessing missing values would distort the results |
| Build one pandas table, then use `groupby` | Fast and simple; no slow row-by-row loops |
| Count everything during generation | Lets the notebook prove its own results are correct |
| Fixed random seed (`SEED = 42`) | Re-running gives the same data (the month in the file names still changes each month) |

**If the data were much bigger**, the same logic would move to tools built for large data, such as reading files in parallel, DuckDB or Spark. The calculations themselves wouldn't change.

---

