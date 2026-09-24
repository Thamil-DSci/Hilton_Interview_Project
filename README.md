# Task 1: Model Benchmarking & A/B Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thamil-DSci/int1_Thamil/blob/main/Thamil_ML.ipynb)

## Steps 

### 1. Problem Selection
1. **Dataset & Problem Statement:**
   * **Dataset:** https://drive.google.com/file/d/10kj_aIX4zpiv2wLFU5KyVUz7yr7V-jYD/view?usp=drive_link
   * **Problem Category:** Supervised Classification
   * **Target Variable:** `Attrition_Flag`
   * **Problem Description:** The Rich-Man bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.Customers leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and reason for same - so that bank could improve upon those areas

   * **Objective:** For the Hilton Data scientist Interview evaluation I'm using the data from Kaggle to compare my Model vs. Kaushik, this is just to show case only and not to compare Kaushik's skills or share my work in Public. https://www.kaggle.com/code/kaushikmajumder/credit-card-customer-churn-prediction/notebook]

---

### 2. Solution Implementation (My Solution: https://github.com/Thamil-DSci/int1_Thamil/blob/main/Hilton_ML_Project.ipynb)
1. **Preprocessing & Pipeline:**
   * Handled missing values using
     * I'm using KNN imputer to impute missing values.
     * `KNNImputer`: Each sample's missing values are imputed by looking at the n_neighbors nearest neighbors found in the training set. Default value for n_neighbors=5.
     * KNN imputer replaces missing values using the average of k nearest non-missing feature values.
     * Nearest points are found based on euclidean distance.
 
   * Scaled features using `Scaled features using StandardScaler (zero mean, unit variance), fitted on the training data only and applied inside a scikit-learn Pipeline, so the same transformation is reused on the test data without leakage.`
    
   * Encoded categorical variables using `Encoded categorical variables using one-hot encoding (pd.get_dummies, drop_first=True). "Unknown" values were first treated as missing, label-encoded and filled with a KNN imputer (k=5), then mapped back to their category labels before one-hot encoding.`
   * Data Split:

      Train / test = 70 / 30, stratified on Attrition_Flag, random_state=7:
       train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

2. **Modeling Technique:**
   * **Selected Algorithm:** `XGBoost (XGBClassifier), tuned with RandomizedSearchCV (50 candidates, 5-fold CV, scored on recall), in a pipeline after StandardScaler
Best parameters: n_estimators=50, learning_rate=0.2, scale_pos_weight=10, subsample=0.7`
   * **Rationale:**
   * Simpler models could not catch enough churners. Logistic regression reached only 0.43 test recall, missing more than half of the churners. Oversampling or undersampling raised it to about 0.79–0.80, but precision fell to about 0.45. A single decision tree reached 0.75–0.79. The churn signal is non-linear (transaction count, amount and revolving balance interact), which a linear model can't capture well.
   * XGBoost had the best recall of all seven algorithms. After tuning, it caught 94.3% of churners with 82.9% precision and 96.0% accuracy. The other ensembles were more precise (0.93) but missed 12–25% of churners (recall 0.75–0.88).
   * RandomizedSearchCV over GridSearchCV: the grid-searched XGBoost gained only 0.8 recall points (about 4 more churners) but lost 10.5 precision points (about 82 more false alarms). The randomized search found the better balance in a fraction of the time.
   * The imbalance is handled inside the model. scale_pos_weight=10 makes each churner count about 10× in the loss, instead of discarding data (undersampling) or creating synthetic rows (SMOTE).
   * Feature strategy: all 19 features were kept, because tree models can use features with weak linear correlation through non-linear effects. "Unknown" categories were KNN-imputed and categoricals one-hot encoded.
---

### 3. Baseline for Comparison
1. **Kaggle Baseline Source:** https://www.kaggle.com/code/kaushikmajumder/credit-card-customer-churn-prediction/notebook
2. **Baseline Overview:** Baseline Overview

Source: Kaushik Majumder, Credit Card Customer Churn Prediction (Kaggle). It uses the same dataset (BankChurners, 10,127 customers, 16.1% churn) and the same business goal: maximise recall on churners, because a missed churner is the costly error.

Data split:

Train / validation / test = 60 / 20 / 20 (6,075 / 2,026 / 2,026 rows), stratified, random_state=1.
Models were compared on the validation set, and the test set was used once, for the final model.

Preprocessing & Pipeline:

Dropped columns:
CLIENTNUM (an ID);
the two leakage columns Naive_Bayes_Classifier_*;
five features judged "uncorrelated" with churn in the correlation heatmap: Customer_Age, Dependent_count, Months_on_book, Credit_Limit and Avg_Open_To_Buy.
That leaves 14 raw features.

Handled missing values: "Unknown" (and the junk value abc in Income_Category) was kept as its own category, Unknown, through custom FillUnknown and CustomValueMasker transformers. Nothing was imputed.

Scaled features: RobustScaler (scales by the IQR, so it's less affected by outliers), fitted on training data only and applied to the 9 numeric columns.
Encoded categorical variables: one-hot encoding (pd.get_dummies, drop_first=True), for 27 features in total.

Class imbalance: random under-sampling of the training set. The non-churners were cut from 5,099 to 976 to match the 976 churners, so the final model trained on 1,952 rows.
Deployment: the steps were later packaged into a scikit-learn Pipeline (with ColumnTransformer and OneHotEncoder) for production use.

Modeling Technique:

Model search:
7 algorithms: Bagging, Random Forest, Gradient Boosting, AdaBoost, XGBoost, Decision Tree and LightGBM.
3 versions of the training data: original, SMOTE-oversampled and randomly under-sampled.

Evaluation: 10-fold stratified cross-validation, scored on recall.

Tuning: the top 4 under-sampled models (XGBoost, AdaBoost, LightGBM, GBM) were tuned with RandomizedSearchCV (50 candidates, 10-fold, recall).

Selected Algorithm: GradientBoostingClassifier, trained on under-sampled data. Parameters: n_estimators=700, max_depth=25, min_samples_leaf=15, min_samples_split=2, max_features='auto' (sqrt).

Author's rationale:
The under-sampled XGBoost was rejected. 

It had the highest validation recall (~99%), but its accuracy (~73%) was below the 84% of a model that simply predicts "no churn" for everyone.

GBM was chosen as the best balance: validation recall ~97%, accuracy ~94%, precision ~74%, ROC-AUC ~0.99, and CV recall 96%, with no sign of bias or variance according to the author.

---

## 4. A/B Testing

### Hypothesis
- **Null Hypothesis ($H_0$):** There is no significant difference in F1 score between the proposed solution (XGBoost) and the baseline (Kaggle Gradient Boosting): $\mu_{proposed} = \mu_{baseline}$.
- **Alternative Hypothesis ($H_1$):** The proposed solution achieves a significantly higher F1 score than the baseline: $\mu_{proposed} > \mu_{baseline}$.
- **Guardrail:** the proposed solution's recall must not be more than **2 percentage points** below the baseline's. The bank cannot accept missing more churners in exchange for fewer false alarms.

### Evaluation Metrics
- **Primary Metric:** `F1 Score` (churn class, threshold 0.5).
  - Both solutions were tuned for recall and already catch ~95% of churners, so recall alone cannot separate them.
  - F1 balances **recall** (a missed churner is a lost customer) and **precision** (a false alarm is a wasted retention offer).
- **Secondary Metrics:** `Recall` (guardrail), `Precision`, `PR-AUC` (threshold-free and suited to imbalanced data), `ROC-AUC`, `Accuracy`.
  - Accuracy is reported but not relied on: with 16% churn, predicting "no churn" for everyone already scores 84%.
    
|model|F1|Recall|Precision|PR\_AUC|ROC\_AUC|Accuracy|
|---|---|---|---|---|---|---|
|Kaggle|0\.8477|0\.9572|0\.7609|0\.9431|0\.9888|0\.9447|
|Mine|0\.8862|0\.9527|0\.8286|0\.9654|0\.9926|0\.9606|

### Data Splitting & Validation
- **Stratified 5-fold cross-validation, repeated 5 times** with a fixed `random_state=1`: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)` on all 10,127 customers, giving **25 paired folds**.
- **Identical splits:** in every fold both models get exactly the same rows, about 8,101 for training and 2,026 for testing (~325 churners).
- **Each model runs its full pipeline inside every training fold**, so there is no leakage:
  - **Proposed:** KNN imputation → one-hot encoding → StandardScaler → XGBoost, trained on all ~8,101 rows with `scale_pos_weight=10`.
  - **Baseline:** drop 5 columns → one-hot encoding → RobustScaler → random under-sampling (~2,600 balanced rows) → Gradient Boosting.
- **Why not compare the published scores:** the two notebooks used different splits (70/30 vs 60/20/20), so their reported test scores are not directly comparable.

## 5. Comparison and Quantitative Results

### Performance Comparison Table
Mean over 25 paired folds.

| Metric | Baseline Solution (Kaggle GBM) | Proposed Solution (Mine-XGBoost) | Delta ($\Delta$) | % Improvement | 95% CI of $\Delta$ | Proposed better in |
|---|---|---|---|---|---|---|
| **F1 (primary)** | 0.848 | **0.886** | **+0.0385** | **+4.5%** | [+0.0243, +0.0526] | 25/25 folds |
| Recall (guardrail) | 0.957 | 0.953 | −0.0045 | −0.5% | [−0.0189, +0.0098] | 6/25 folds |
| Precision | 0.761 | **0.829** | +0.0677 | +8.9% | [+0.0458, +0.0896] | 25/25 folds |
| PR-AUC | 0.943 | **0.965** | +0.0223 | +2.4% | [+0.0102, +0.0345] | 25/25 folds |
| ROC-AUC | 0.989 | **0.993** | +0.0038 | +0.4% | [+0.0018, +0.0057] | 25/25 folds |
| Accuracy | 0.945 | **0.961** | +0.0159 | +1.7% | [+0.0104, +0.0214] | 25/25 folds |

**In customer terms (per fold of 2,026 customers, ~325 churners):** both models catch about 310 churners, but the proposed solution raises **~64 false alarms against ~98 for the baseline, about 35% fewer**.

### Statistical Significance Test
- **Test Type:** one-sided **paired t-test** on the 25 per-fold differences, with the **Nadeau–Bengio correction** for overlapping CV training sets. A plain t-test would overstate significance.
- **p-value:** **4.5 × 10⁻⁶** for F1 ($\alpha = 0.05$).
- **Result:** **Statistically significant.** $H_0$ is rejected: the proposed solution has a higher F1 score.
- **Guardrail:** **passed.** The recall difference is not significant (−0.45 pts, p = 0.74), and the lower 95% CI bound (−1.89 pts) stays inside the −2 pt margin.
- **Other metrics:** precision, PR-AUC, ROC-AUC and accuracy are also significantly higher (all p < 0.001, better in 25/25 folds).

![A/B test results](images/ab_results.png)
*Left: mean difference per metric with 95% CI (dotted line = −2 pt recall margin). Middle/right: F1 and precision per fold; each grey line joins the same fold.*

## 6. Analysis and Explanation

<img width="1589" height="393" alt="image" src="https://github.com/user-attachments/assets/9da9b4fb-4067-48dc-b5d8-499674ac6b61" />


### Insights on Differences
The proposed solution catches **the same churners** as the baseline but with **far fewer false alarms**. Its precision–recall curve is higher (PR-AUC 0.965 vs 0.943), so this is better ranking of customers, not just a different threshold. The main reasons:

1. **Keeping all the data vs under-sampling.**
   - The baseline discards ~70% of the non-churners in each training fold (8,101 → ~2,600 rows) to balance the classes. That pushes its recall up, but the model sees far fewer examples of loyal customers, so it over-flags them.
   - The proposed solution handles the imbalance with **class weighting** (`scale_pos_weight=10`) and learns from every row.
2. **Keeping all features.**
   - The baseline dropped `Customer_Age`, `Dependent_count`, `Months_on_book`, `Credit_Limit` and `Avg_Open_To_Buy` because their *linear* correlation with churn was near zero.
   - Tree models use non-linear effects and interactions, so weak linear correlation is not a good reason to drop a feature.
   - The customers that only the baseline wrongly flags tend to be younger and newer to the bank, which is exactly the information it removed.
3. **Model capacity and regularisation.**
   - The baseline uses 700 trees of depth 25 fitted to ~2,600 rows, which tends to memorise the training data.
   - The proposed XGBoost (50 trees, `gamma=5`, `subsample=0.7`) is smaller and regularised, and generalises more steadily: its precision varies less from fold to fold.

### Strengths vs. Weaknesses
**Strengths**
- ~35% fewer wasted retention offers at the same recall, with significantly higher F1, precision, PR-AUC and accuracy in **all 25 folds**.
- Uses all customers and all features; no training data is discarded.
- Smaller and faster model: 50 trees vs 700 deep trees, so training and scoring are cheaper.
- More stable across folds.

**Weaknesses**
- **Recall is not better.** It is slightly lower on average (−0.45 pts, not significant), and the CI lower bound (−1.89 pts) is close to the 2-pt margin, so a small recall loss cannot be fully ruled out.
- **KNN imputation adds a step at scoring time** (it needs the training data to find neighbours) and treats category codes as numbers, which is a heuristic for nominal variables.
- **Mild over-fitting:** train recall is 1.00 against ~0.95 on unseen data.
- **Tuned on recall only, with the default 0.5 threshold** rather than a threshold chosen from business costs.

### Gaps & Future Improvements
1. **Choose the threshold from business costs.** Pick the decision threshold (inside CV) that minimises *(cost of a lost customer × missed churners) + (cost of an offer × false alarms)*, or that hits a target recall such as ≥ 0.96. Because the proposed model ranks customers better, it can match the baseline's recall while keeping higher precision.
2. **Tune on PR-AUC or F2 instead of raw recall**, so the search sees the cost of false alarms.
3. **Simplify the categorical handling:** keep "Unknown" as its own category, or use native categorical support (LightGBM/CatBoost), instead of KNN-imputing label codes.
4. **Feature engineering:** e.g. average transaction value (`Total_Trans_Amt / Total_Trans_Ct`), recent-activity drop (Q4/Q1 change × months inactive).
5. **Use nested cross-validation for tuning**, so the reported score is not measured on the same data used to choose the model.
6. **Explainability:** SHAP values per customer, so the retention team sees *why* someone is flagged.
7. **Run an online A/B test:** randomise flagged customers into offer / no-offer groups and measure **customers actually retained**. Offline metrics are only a proxy for business impact.
