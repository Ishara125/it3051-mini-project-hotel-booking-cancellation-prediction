# Evaluation 2 – Comprehensive Viva Study Guide

**Course:** IT3051 – Fundamentals of Data Mining  
**Mini Project:** Hotel Booking Cancellation Prediction  
**Assessment:** Progress Evaluation 2 (30% Individual Viva)  
**Target Variable:** `is_canceled` (0 = Not Canceled / Honored, 1 = Canceled)

---

## 1. Top 20 Fundamental Viva Questions & Clear Answers

### Q1: What is classification?
**Answer:** Classification is a supervised machine learning task where the model learns from historical data with known ground-truth labels to predict a discrete categorical class (in our project, a binary outcome: $0$ or $1$) for new, unseen instances.

### Q2: What is the target variable in this project?
**Answer:** The target variable is `is_canceled`. It is binary:
- **0 (Negative class):** The booking was honored (guest arrived / checked in).
- **1 (Positive class):** The booking was canceled prior to arrival.

### Q3: Why did we implement four different models?
**Answer:** To compare diverse algorithmic paradigms—a linear model (**Logistic Regression**), a single tree (**Decision Tree**), a bagging ensemble (**Random Forest**), and a boosting ensemble (**Gradient Boosting**). Comparing different paradigms shows how linear vs. non-linear and bagging vs. boosting architectures handle complex tabular data, non-linear interactions, and class imbalance.

### Q4: What is a baseline model?
**Answer:** A baseline model is an initial model trained with default, un-optimized hyperparameters. It acts as an empirical performance floor against which all subsequent tuning, feature engineering, and optimizations are measured.

### Q5: What is hyperparameter tuning?
**Answer:** Hyperparameters are configuration settings defined *before* training that control how the algorithm learns (e.g., `max_depth`, `learning_rate`, `C`). Tuning is the systematic process of searching for the combination of hyperparameters that maximizes generalization performance on validation data without overfitting.

### Q6: What is cross-validation (CV)?
**Answer:** Cross-validation is a resampling technique where the training dataset is partitioned into $K$ equal subsets (folds). The model is trained on $K-1$ folds and validated on the remaining fold, rotating $K$ times. The average validation score gives an unbiased estimate of generalization performance.

### Q7: What is StratifiedKFold and why did we use it?
**Answer:** In standard K-Fold, random splitting might create folds with unequal class ratios. **StratifiedKFold** guarantees that each fold contains exactly the same proportion of classes ($27.52\%$ cancellations) as the full training set, which is crucial for imbalanced data.

### Q8: Why must all models be evaluated on the exact same test set?
**Answer:** To ensure scientific fairness. If models were evaluated on different test subsets, differences in accuracy could simply be due to one test set having easier or harder samples. An identical test set ensures performance differences reflect genuine algorithmic capability.

### Q9: Why must test data be strictly excluded from tuning and preprocessing?
**Answer:** To prevent **information leakage (data snooping)**. If the test set is used to tune hyperparameters or calculate scaling parameters (mean/std), test patterns leak into the model, producing overly optimistic performance estimates that fail in real-world deployment.

### Q10: What is Accuracy, and what is its formula?
**Answer:** Accuracy is the fraction of total predictions that were correct:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
In our holdout test set, tuned Gradient Boosting achieved **85.01%** accuracy.

### Q11: What is Precision, and what does it mean here?
**Answer:** Precision measures: *Of all bookings the model flagged as "Canceled", how many were actually canceled?*
$$\text{Precision} = \frac{TP}{TP + FP}$$
Tuned Gradient Boosting achieved **75.65%** precision.

### Q12: What is Recall (Sensitivity), and what does it mean here?
**Answer:** Recall measures: *Of all actual cancellations that occurred, how many did the model successfully catch?*
$$\text{Recall} = \frac{TP}{TP + FN}$$
Tuned Gradient Boosting caught **67.14%** (3,224 out of 4,802 cancellations).

### Q13: What is the F1-Score?
**Answer:** The $F_1$-score is the harmonic mean of Precision and Recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It provides a single balanced metric that penalizes extreme imbalances between Precision and Recall. Tuned Gradient Boosting achieved **0.7114**.

### Q14: What is ROC-AUC?
**Answer:** **Receiver Operating Characteristic - Area Under the Curve (ROC-AUC)** measures the model's ability to rank positive cases higher than negative cases across *all* possible classification probability thresholds ($0.0$ to $1.0$).
- $0.5$ = Random guessing
- $1.0$ = Perfect discrimination
- Tuned Gradient Boosting achieved **0.9152** (outstanding discrimination).

### Q15: What is overfitting, and how did we measure it?
**Answer:** Overfitting occurs when a model memorizes random noise and idiosyncrasies in the training data rather than learning true underlying patterns, leading to poor test performance.
We measured it as the **Overfitting Gap**:
$$\text{Overfitting Gap} = \text{Train Accuracy} - \text{Test Accuracy}$$
For example, baseline Decision Tree had a massive $19.01\%$ gap ($99.75\% - 80.73\%$), which tuning reduced to $2.75\%$.

### Q16: What is a Confusion Matrix?
**Answer:** A $2 \times 2$ table cross-tabulating actual vs. predicted labels:
- **True Positive (TP):** Actually canceled, predicted canceled.
- **True Negative (TN):** Actually honored, predicted honored.
- **False Positive (FP / Type I Error):** Actually honored, incorrectly predicted canceled.
- **False Negative (FN / Type II Error):** Actually canceled, missed by model (costliest error for hotel).

### Q17: What is data leakage?
**Answer:** Data leakage occurs when information from outside the training environment (or post-event future information) is inadvertently introduced into the model training pipeline, creating deceptively high but invalid performance.

### Q18: How did we prevent data leakage in this project?
**Answer:**
1. **Target Leakage Removal:** Dropped `reservation_status` and `reservation_status_date` (which directly state if the booking was "Check-Out" or "Canceled" after the fact).
2. **Preprocessing Isolation:** All imputers, standard scalers, and one-hot encoders were fitted **strictly on `X_train`** and only applied (transformed) to `X_test`.
3. **Cross-Validation Integrity:** Hyperparameter searches used 5-fold CV exclusively within `X_train`.

### Q19: Why is Accuracy alone insufficient for this problem?
**Answer:** Our test set has a **72.48% : 27.52%** class imbalance (12,644 non-cancellations vs. 4,802 cancellations). A naive dummy model that always predicts "Not Canceled" achieves **72.48% accuracy** while catching **0%** of cancellations. Accuracy conceals high False Negatives. In hospitality, an undetected cancellation leaves a room permanently empty and perishable revenue lost. Therefore, **$F_1$-score, Recall, and ROC-AUC** are essential.

### Q20: Why was Tuned Gradient Boosting selected as the final model?
**Answer:** Tuned Gradient Boosting outperformed all other models across the holistic trade-off:
- **Highest ROC-AUC (0.9152):** Unmatched capability to rank cancellation risk across probability thresholds.
- **Highest Test Accuracy (85.01%):** Correct on 14,831 out of 17,446 holdout test bookings.
- **Highest F1-Score (0.7114):** Best balance between precision ($75.65\%$) and recall ($67.14\%$).
- **Virtually Zero Overfitting (0.0062):** Train accuracy was $85.62\%$ vs. test accuracy of $85.01\%$, demonstrating robust generalization.

---

## 2. Verified Summary of the Four Algorithms

| Model | Baseline Test Acc | Tuned Test Acc | Tuned Precision | Tuned Recall | Tuned F1 | Tuned ROC-AUC | Best CV F1 | Overfitting Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Logistic Regression** | 0.8158 | 0.7863 | 0.5772 | 0.8357 | 0.6828 | 0.8776 | 0.6719 | **-0.0031** |
| **2. Decision Tree** | 0.8073 | 0.7974 | 0.5895 | **0.8696** | 0.7027 | 0.8783 | 0.6994 | 0.0275 |
| **3. Random Forest** | **0.8501** | 0.7835 | 0.5716 | 0.8521 | 0.6842 | 0.8909 | 0.6822 | 0.0086 |
| **4. Gradient Boosting** | 0.8394 | **0.8501** | **0.7565** | 0.6714 | **0.7114** | **0.9152** | **0.7076** | 0.0062 |

---

## 3. Individual Algorithm Quick Reference for Members

### Member 1: Logistic Regression
- **Type:** Parametric linear classifier with sigmoid activation.
- **Optimization:** `GridSearchCV` (30 combinations, 5 folds = 150 fits).
- **Best Hyperparameters:** `C=1.0`, `penalty='l2'`, `solver='liblinear'`, `class_weight='balanced'`.
- **Key Insight:** `class_weight='balanced'` boosted Recall dramatically from $56.66\%$ to $83.57\%$ by shifting the decision boundary to protect against missed cancellations.

### Member 2: Decision Tree
- **Type:** Non-parametric hierarchical splitting tree using Gini impurity.
- **Optimization:** `GridSearchCV` (252 combinations, 5 folds = 1,260 fits).
- **Best Hyperparameters:** `criterion='gini'`, `max_depth=15`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight='balanced'`.
- **Key Insight:** Baseline tree severely overfitted (19.01% gap). Setting `max_depth=15` pruned spurious branches and cut the gap to $2.75\%$, achieving the highest recall ($86.96\%$).

### Member 3: Random Forest
- **Type:** Bagging ensemble of 150 decorrelated decision trees with feature subsampling (`sqrt`).
- **Optimization:** `RandomizedSearchCV` (15 iterations, 5 folds = 75 fits).
- **Best Hyperparameters:** `n_estimators=150`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=4`, `max_features='sqrt'`, `class_weight='balanced'`, `bootstrap=True`.
- **Key Insight:** Capped tree depth and balanced class weights to eliminate a $14.74\%$ baseline overfitting gap down to $0.86\%$, raising Recall from $63.04\%$ to $85.21\%$.

### Member 4: Gradient Boosting
- **Type:** Sequential boosting ensemble fitting shallow trees to loss pseudo-residuals.
- **Optimization:** `RandomizedSearchCV` (12 iterations, 5 folds = 60 fits).
- **Best Hyperparameters:** `n_estimators=200`, `learning_rate=0.15`, `max_depth=4`, `min_samples_split=10`, `min_samples_leaf=4`, `subsample=0.9`, `max_features=None`.
- **Key Insight:** Selected as the final champion model. Provides the highest overall discrimination (ROC-AUC 0.9152) and maintains high precision (75.65%) while controlling overfitting (gap 0.0062).
