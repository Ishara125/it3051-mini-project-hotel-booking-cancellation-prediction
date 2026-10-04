# Member 3 – Random Forest Classifier: Viva Preparation Guide

**Project:** Hotel Booking Cancellation Prediction  
**Target Variable:** `is_canceled` (0 = Not Canceled / Honored, 1 = Canceled)  
**Algorithm:** Random Forest Classifier (Bagging Ensemble)  
**Evaluation:** Progress Evaluation 2 (30%)

---

## 1. Core Machine Learning Concepts (Simplified)

### 1.1 What is a Random Forest?
A **Random Forest** is an ensemble algorithm made up of many individual Decision Trees (in our project, 150 trees). Each tree is trained on a slightly different version of the dataset and makes its own prediction. The forest then combines all the individual tree predictions through **majority voting** (for classification) or probability averaging.

### 1.2 What is Ensemble Learning?
Ensemble learning is a machine learning paradigm where multiple models (called "base learners" or "weak learners") are combined together to produce a stronger, more accurate, and more robust predictor than any single model alone. It follows the principle of "the wisdom of the crowd."

### 1.3 What is Bagging (Bootstrap Aggregating)?
**Bagging** is the specific ensemble technique used by Random Forest:
1. **Bootstrap**: Multiple random subsets of the training data are sampled **with replacement**.
2. **Aggregating**: An independent tree is trained on each bootstrap sample, and their predictions are aggregated (averaged or voted) at test time.
- Bagging primarily **reduces variance** (overfitting) without increasing bias.

### 1.4 What is Bootstrap Sampling?
Sampling **with replacement** means each time a record is randomly picked from the training set, it is placed back so it can be picked again.
- For a dataset of size $N$, a bootstrap sample typically contains approximately $63.2\%$ unique records, leaving about $36.8\%$ unselected (known as the **Out-Of-Bag / OOB** samples, which can serve as a built-in validation set).

### 1.5 Difference Between Decision Tree and Random Forest
| Dimension | Decision Tree (Member 2) | Random Forest (Member 3) |
| :--- | :--- | :--- |
| **Model Type** | Single individual tree | Ensemble of 150 trees |
| **Variance / Overfitting** | High variance; prone to memorizing noise | Low variance; averaging decorrelates errors |
| **Split Candidate Features** | Evaluates all features at every split | Evaluates only a random subset ($\sqrt{p}$) at each split |
| **Interpretability** | Easy to visualize a single flowchart | "Black box"; interpreted via feature importance |
| **Overfitting Gap in Our Baseline** | Severe gap: $19.01\%$ (Train $99.75\%$ vs Test $80.73\%$) | High gap: $14.74\%$ (Train $99.75\%$ vs Test $85.01\%$) |

### 1.6 Difference Between Random Forest (Bagging) and Gradient Boosting (Boosting)
| Dimension | Random Forest (Bagging) | Gradient Boosting (Boosting) |
| :--- | :--- | :--- |
| **Tree Construction** | **Parallel & Independent**: All trees are built simultaneously | **Sequential**: Each tree is built one after another |
| **Tree Depth** | Deep, unpruned or semi-pruned trees (low bias, high variance) | Shallow trees / "stumps" (high bias, low variance) |
| **Objective of New Trees** | Fit independent bootstrap samples | Fit the residual errors (gradient) of previous trees |
| **Weights / Voting** | Equal voting weight for all trees | Weighted combination based on learning rate $\eta$ |
| **Goal** | Primarily reduces **variance** | Reduces **both bias and variance** |

---

## 2. Experimental Results: Baseline vs. Tuned Random Forest

All evaluations were conducted on the group's untouched holdout test partition ($17,446$ records, $4,802$ cancellations, $27.52\%$ cancellation rate).

### 2.1 Verified Metrics Table
| Metric | Baseline Random Forest | Tuned Random Forest | Change / Direction |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.8501** ($85.01\%$) | 0.7835 ($78.35\%$) | $-6.66\%$ |
| **Precision** | **0.7826** ($78.26\%$) | 0.5716 ($57.16\%$) | $-21.10\%$ |
| **Recall** | 0.6304 ($63.04\%$) | **0.8521** ($85.21\%$) | **$+22.17\%$ (Major gain)** |
| **F1-Score** | **0.6983** ($0.6983$) | 0.6842 ($0.6842$) | $-0.0141$ |
| **ROC-AUC** | **0.9059** ($0.9059$) | 0.8909 ($0.8909$) | $-0.0150$ |
| **Train Accuracy** | 0.9975 ($99.75\%$) | 0.7921 ($79.21\%$) | $-20.54\%$ |
| **Test Accuracy** | 0.8501 ($85.01\%$) | 0.7835 ($78.35\%$) | $-6.66\%$ |
| **Overfitting Gap** | **0.1474 ($14.74\%$)** | **0.0086 ($0.86\%$)** | **Virtually eliminated!** |
| **Best 5-Fold CV F1** | N/A | **0.6822** | Robust cross-validation |

---

## 3. Best Hyperparameters Explained Simply

Search method used: `RandomizedSearchCV` (15 iterations across 5 stratified folds = 75 total fits, optimized for $F_1$-score on training data only).

1. `n_estimators = 150`
   - **What it means:** The number of trees built in the forest.
   - **Why:** 150 trees provide smooth ensemble averaging and stable predictions without excessive computation time.
2. `max_depth = 20`
   - **What it means:** The maximum length from the root node to the deepest leaf node.
   - **Why:** Prevents trees from growing infinitely deep (which caused the $99.75\%$ memorization in the baseline).
3. `min_samples_split = 5`
   - **What it means:** A node must contain at least 5 samples before it is permitted to split further.
   - **Why:** Prevents the model from creating splits tailored to tiny, noisy clusters of records.
4. `min_samples_leaf = 4`
   - **What it means:** Every final leaf node must contain at least 4 training observations.
   - **Why:** Smooths leaf-node probability estimates and regularizes against individual outlier records.
5. `max_features = 'sqrt'`
   - **What it means:** At every node split, only $\sqrt{p}$ features (approximately $\sqrt{899} \approx 30$ features) are randomly evaluated.
   - **Why:** Decorrelates the trees so that strong features don't dominate every single tree.
6. `class_weight = 'balanced'`
   - **What it means:** Inversely weights class frequencies ($w_1 = \frac{N}{2 \cdot N_1} \approx 1.82$).
   - **Why:** Forces the model to penalize misclassifying cancellations, directly driving the massive boost in Recall ($63.04\% \to 85.21\%$).
7. `bootstrap = True`
   - **What it means:** Each tree is trained on a random bootstrap sample with replacement.
   - **Why:** Ensures sample diversity across all 150 trees.

---

## 4. The Critical Tuning Trade-Off (How to Defend Your Model in Viva)

> **Key Viva Talking Point:**  
> *"Tuning did NOT simply make every single metric higher. Instead, hyperparameter tuning resolved two major business and statistical issues: it **eliminated severe overfitting** and **dramatically boosted Recall**."*

### Why this trade-off happened:
1. **Severe Overfitting Cured:**
   - In baseline RF, `Train Accuracy = 99.75%` while `Test Accuracy = 85.01%` (an overfitting gap of $14.74\%$).
   - By constraining `max_depth=20`, `min_samples_leaf=4`, and `min_samples_split=5`, the model was regularized. The gap collapsed to **$0.86\%$** ($79.21\%$ train vs $78.35\%$ test).
2. **Prioritizing Cancellation Detection (Business Need):**
   - The hotel loses substantial revenue when a cancellation is missed (False Negative).
   - Setting `class_weight='balanced'` forced the decision boundaries to favor the minority cancellation class.
   - As a result, **Recall surged from $63.04\%$ to $85.21\%$** (catching 4,092 out of 4,802 cancellations).
   - The cost of catching more cancellations is more False Positives (predicting a cancellation when the guest actually shows up), which lowers **Precision** ($78.26\% \to 57.16\%$) and overall **Accuracy** ($85.01\% \to 78.35\%$).
   - For hotel management, a high-recall early warning system allows proactive overbooking and re-confirmation campaigns.

---

## 5. 15 Likely Viva Questions & Concise Answers for Member 3

#### Q1: What was your specific algorithm in this project?
**Answer:** I was responsible for Member 3 – the **Random Forest Classifier**, an ensemble bagging algorithm that combines 150 decorrelated decision trees using majority voting.

#### Q2: Why did you use `RandomizedSearchCV` instead of `GridSearchCV`?
**Answer:** Because our preprocessed feature matrix has **899 features** and **69,782 training samples**. An exhaustive grid search across combinations of trees would require thousands of fits and take hours or days. `RandomizedSearchCV` evaluated 15 well-distributed configurations across 5 folds (75 fits) efficiently within a manageable compute budget.

#### Q3: How did you ensure your tuning did not cause data leakage?
**Answer:** The tuning was executed strictly on `X_train` using 5-fold Stratified Cross-Validation. The holdout test set (`X_test`, 17,446 records) was kept completely untouched and was only evaluated once on the final tuned model.

#### Q4: Why did baseline Random Forest overfit, and how did you detect it?
**Answer:** The baseline tree depth was unconstrained (`max_depth=None`, `min_samples_leaf=1`), allowing trees to grow until almost every leaf was pure. It scored $99.75\%$ on training data but only $85.01\%$ on test data—a gap of $14.74\%$.

#### Q5: How did you fix that overfitting?
**Answer:** I regularized the trees by capping `max_depth=20`, raising `min_samples_leaf=4`, and `min_samples_split=5`. This reduced the overfitting gap to just $0.86\%$.

#### Q6: Why did Accuracy drop from $85.01\%$ to $78.35\%$ after tuning?
**Answer:** Because `class_weight='balanced'` was introduced. When you penalize missed cancellations more heavily, the model predicts "Canceled" more aggressively. This increases False Positives, lowering raw Accuracy and Precision, but significantly raises **Recall from $63.04\%$ to $85.21\%$**.

#### Q7: In the hotel context, is high Recall or high Precision more important?
**Answer:** **Recall** is generally more critical because an undetected cancellation results in an empty room and $100\%$ perishable revenue loss. A false alarm (False Positive) only triggers a re-confirmation email or modest retention incentive.

#### Q8: What does `max_features='sqrt'` do in Random Forest?
**Answer:** At each node split, it restricts the tree to randomly consider only $\sqrt{p} \approx 30$ features out of 899. This decorrelates the trees, preventing a few dominant features (like `lead_time` or `deposit_type`) from making all trees identical.

#### Q9: What metric did you optimize during tuning?
**Answer:** I optimized for **$F_1$-score** rather than Accuracy. Because the dataset has a $72.5\% : 27.5\%$ class imbalance, optimizing Accuracy encourages the model to simply predict the majority class. $F_1$-score balances precision and recall.

#### Q10: What were your best hyperparameters?
**Answer:** `n_estimators=150`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=4`, `max_features='sqrt'`, `class_weight='balanced'`, and `bootstrap=True`.

#### Q11: What was your best Cross-Validation score?
**Answer:** The best 5-fold cross-validation $F_1$-score on the training set was **0.6822** (which closely mirrors the holdout test $F_1$ of **0.6842**, proving solid generalization).

#### Q12: How does Random Forest calculate feature importance?
**Answer:** By measuring the **mean decrease in impurity (MDI / Gini importance)** across all 150 trees whenever a given feature is used to split a node, weighted by the number of samples reaching that node.

#### Q13: What top features did Random Forest identify?
**Answer:** `lead_time`, `deposit_type_Non Refund`, `adr` (average daily rate), `country_PRT`, `total_stay`, and `previous_cancellations`.

#### Q14: How does your Random Forest compare to Member 2's Decision Tree?
**Answer:** Member 2's baseline tree had a massive $19.01\%$ overfitting gap and lower ROC-AUC ($0.7625$). Random Forest averaged 150 trees, achieving a much higher baseline ROC-AUC of **0.9059**, showing the immense power of ensemble averaging.

#### Q15: Why was Gradient Boosting selected over your Random Forest as the final model?
**Answer:** While tuned Random Forest achieved high recall ($85.21\%$), its precision dropped to $57.16\%$. **Gradient Boosting** achieved a superior balance: higher overall Test Accuracy ($85.01\%$), higher Precision ($75.65\%$), higher $F_1$ ($0.7114$), the highest ROC-AUC ($0.9152$), and a tiny overfitting gap ($0.0062$).
