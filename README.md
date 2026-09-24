# Task 1: Model Benchmarking & A/B Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thamil-DSci/int1_Thamil/blob/main/Thamil_ML.ipynb)

## Steps to Complete the Task

### 1. Problem Selection
1. **Dataset & Problem Statement:**
   * **Dataset:** https://drive.google.com/file/d/10kj_aIX4zpiv2wLFU5KyVUz7yr7V-jYD/view?usp=drive_link
   * **Problem Category:** Supervised Classification
   * **Target Variable:** `Attrition_Flag`
   * **Problem Description:** The Rich-Man bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.Customers leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and reason for same - so that bank could improve upon those areas

   * ****Objective:** For the Hilton Data scientist Interview evaluation I'm using the data from Kaggle to compare my Model vs. Kaushik, this is just to show case only and not to compare Kaushik's skills or share my work in Public. https://www.kaggle.com/code/kaushikmajumder/credit-card-customer-churn-prediction/notebook]

---

### 2. Solution Implementation (Your Approach)
1. **Preprocessing & Pipeline:**
   * Handled missing values using
     * I'm using KNN imputer to impute missing values.
     * `KNNImputer`: Each sample's missing values are imputed by looking at the n_neighbors nearest neighbors found in the training set. Default value for n_neighbors=5.
     * KNN imputer replaces missing values using the average of k nearest non-missing feature values.
     * Nearest points are found based on euclidean distance.
 
   * Scaled features using `Scaled features using StandardScaler (zero mean, unit variance), fitted on the training data only and applied inside a scikit-learn Pipeline, so the same transformation is reused on the test data without leakage.`
    
   * Encoded categorical variables using `Encoded categorical variables using one-hot encoding (pd.get_dummies, drop_first=True). "Unknown" values were first treated as missing, label-encoded and filled with a KNN imputer (k=5), then mapped back to their category labels before one-hot encoding.`
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
1. **Kaggle Baseline Source:** [Insert Link to Published Kaggle Notebook/Solution][cite: 3]
2. **Baseline Overview:** [Describe the baseline's approach, model type, and preprocessing methods][cite: 3]

---

### 4. A/B Testing Setup
1. **Hypothesis:**
   * **Null Hypothesis ($H_0$):** There is no significant difference in performance between the proposed solution and the baseline model ($\mu_{\text{proposed}} = \mu_{\text{baseline}}$).[cite: 3]
   * **Alternative Hypothesis ($H_1$):** The proposed solution significantly outperforms the baseline model on the primary evaluation metric ($\mu_{\text{proposed}} > \mu_{\text{baseline}}$).[cite: 3]
2. **Evaluation Metrics:**
   * **Primary Metric:** `[e.g., ROC-AUC / RMSE / F1 Score]`[cite: 3]
   * **Secondary Metrics:** `[e.g., Accuracy, Precision, Recall / MAE]`[cite: 3]
3. **Data Splitting & Validation:**
   * Used Stratified 5-Fold Cross-Validation with fixed `random_state` across both models to ensure a fair comparison.[cite: 3]

---

### 5. Comparison and Quantitative Results
1. **Performance Comparison Table:**[cite: 3]

   | Metric | Baseline Solution | Proposed Solution | Delta ($\Delta$) | % Improvement |
   | :--- | :--- | :--- | :--- | :--- |
   | **Primary Metric** | `0.00` | `0.00` | `+0.00` | **+0.0%** |
   | **Metric 2** | `0.00` | `0.00` | `+0.00` | **+0.0%** |
   | **Metric 3** | `0.00` | `0.00` | `+0.00` | **+0.0%** |

2. **Statistical Significance Test:**[cite: 3]
   * **Test Type:** Paired Student's t-test / Bootstrap resampling[cite: 3]
   * **p-value:** `0.00` ($\alpha = 0.05$)[cite: 3]
   * **Result:** [Statistically Significant / Not Significant][cite: 3]

---

### 6. Analysis and Explanation
1. **Insights on Differences:** [Explain why your model performed differently than the baseline][cite: 3]
2. **Strengths vs. Weaknesses:**
   * **Strengths:** [List main advantages of your solution][cite: 3]
   * **Weaknesses:** [List trade-offs like higher compute cost or inference latency][cite: 3]
3. **Gaps & Future Improvements:** [Discuss potential reasons for performance gaps and next steps][cite: 3]
