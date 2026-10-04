"""Reusable helper functions for Member 1 - Logistic Regression.

These utilities handle evaluation, plotting, and reporting so the
notebook stays clean.  Every function is intentionally simple so an
undergraduate student can read and explain each line during a viva.

Member 1 responsibilities (Evaluation 2):
  - Logistic Regression baseline and tuned model
  - Evaluation metrics, confusion matrix, ROC curve
  - Hyperparameter tuning with cross-validation on training data only
  - Overfitting check and model interpretation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# -- colour palette used across all Member 1 plots --------------------------
_PALETTE = {
    "primary": "#4C72B0",   # cool blue  - general bars / lines
    "positive": "#DD8452",  # warm orange - cancelled class
    "negative": "#55A868",  # green - not-cancelled class
    "light_bg": "#F8F9FA",
    "grid": "#E0E0E0",
}


# -- 1.  Comprehensive evaluation -------------------------------------------

def evaluate_classifier(
    y_true,
    y_pred,
    y_prob,
    label: str = "Model",
) -> dict:
    """Calculate and print the full set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 = Not cancelled, 1 = Cancelled).
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted probabilities for the positive class (cancelled = 1).
    label : str
        Display name used in the printed header.

    Returns
    -------
    dict
        Metric name -> float value, easy to place in a comparison table.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    border = "=" * 60
    print(f"\n{border}")
    print(f"  {label}  -  Classification Results")
    print(border)
    print(f"  Accuracy  : {acc:.4f}   (overall proportion correct)")
    print(f"  Precision : {prec:.4f}   (of predicted cancellations, how many are real)")
    print(f"  Recall    : {rec:.4f}   (of real cancellations, how many are caught)")
    print(f"  F1-score  : {f1:.4f}   (harmonic mean of precision & recall)")
    print(f"  ROC-AUC   : {auc:.4f}   (ability to rank cancelled above not-cancelled)")
    print(border)
    print("\nFull classification report:\n")
    print(classification_report(y_true, y_pred, target_names=["Not Cancelled", "Cancelled"]))

    return {
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-score":  round(f1,   4),
        "ROC-AUC":   round(auc,  4),
    }


# -- 2.  Confusion matrix plot -----------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    figsize: tuple = (6, 5),
) -> None:
    """Plot a labelled confusion matrix with plain-English quadrant descriptions.

    Quadrant layout:
        TN = True Negative  (predicted Not Cancelled, actually Not Cancelled)
        FP = False Positive (predicted Cancelled,     actually Not Cancelled)
        FN = False Negative (predicted Not Cancelled, actually Cancelled)
        TP = True Positive  (predicted Cancelled,     actually Cancelled)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    class_names = ["Not Cancelled (0)", "Cancelled (1)"]

    # Annotations: count + short label
    annot = np.array([
        [f"{tn}\nTrue Negative\n(Correct)", f"{fp}\nFalse Positive\n(False alarm)"],
        [f"{fn}\nFalse Negative\n(Missed)",  f"{tp}\nTrue Positive\n(Caught)"],
    ])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=1.5,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("True Label", fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.show()
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


# -- 3.  ROC curve plot -------------------------------------------------------

def plot_roc_curve(
    y_true,
    y_prob,
    label: str = "Logistic Regression",
    figsize: tuple = (7, 5),
) -> None:
    """Plot the ROC curve and shade the area under it.

    The diagonal dashed line represents a random classifier (AUC=0.50).
    The further the curve bows toward the top-left corner, the better
    the model separates cancelled from not-cancelled bookings.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_PALETTE["light_bg"])
    ax.plot(fpr, tpr, color=_PALETTE["primary"], lw=2.5,
            label=f"{label}  (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=_PALETTE["primary"])
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="Random classifier (AUC = 0.50)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve - Logistic Regression", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.8)
    plt.tight_layout()
    plt.show()
    print(f"  ROC-AUC = {auc:.4f}")


# -- 4.  Baseline-vs-tuned comparison table -----------------------------------

def comparison_table(
    baseline_metrics: dict,
    tuned_metrics: dict,
) -> pd.DataFrame:
    """Return a tidy DataFrame comparing baseline and tuned metrics."""
    rows = []
    for metric in ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"):
        b = baseline_metrics[metric]
        t = tuned_metrics[metric]
        change = t - b
        direction = "up" if change > 0.0001 else ("down" if change < -0.0001 else "~same")
        rows.append({
            "Metric":    metric,
            "Baseline":  f"{b:.4f}",
            "Tuned":     f"{t:.4f}",
            "Change":    f"{'+' if change >= 0 else ''}{change:.4f} ({direction})",
        })
    return pd.DataFrame(rows).set_index("Metric")


# -- 5.  Overfitting check ----------------------------------------------------

def overfitting_check(
    model,
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
) -> pd.DataFrame:
    """Compare training vs test accuracy and F1 to detect overfitting.

    A large gap (e.g. >5 pp) suggests the model has memorised the
    training data and may not generalise well.  A small gap is normal.
    """
    y_train_pred = model.predict(X_train_proc)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1  = f1_score(y_train, y_train_pred, zero_division=0)

    y_test_pred = model.predict(X_test_proc)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred, zero_division=0)

    df = pd.DataFrame(
        {
            "Split":    ["Training", "Test", "Gap (Train - Test)"],
            "Accuracy": [f"{train_acc:.4f}", f"{test_acc:.4f}", f"{train_acc - test_acc:.4f}"],
            "F1-score": [f"{train_f1:.4f}",  f"{test_f1:.4f}",  f"{train_f1  - test_f1:.4f}"],
        }
    ).set_index("Split")
    return df


# -- 6.  Coefficient interpretation -------------------------------------------

def coefficient_table(
    model,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and rank Logistic Regression coefficients.

    A positive coefficient means the feature is associated with a
    HIGHER probability of cancellation (is_canceled = 1).
    A negative coefficient means the feature is associated with a
    LOWER probability of cancellation.

    IMPORTANT: These are model associations, not proof of causation.

    Parameters
    ----------
    model : fitted LogisticRegression
    feature_names : list[str]
        Ordered list of processed feature names.
    top_n : int
        Number of top-positive and top-negative features to return.

    Returns
    -------
    pd.DataFrame
        Sorted coefficient table (most positive first).
    """
    if hasattr(model, "named_steps"):
        # sklearn Pipeline - get the LR step
        lr = model.named_steps.get("classifier") or model.named_steps.get("lr")
    else:
        lr = model

    coef = lr.coef_.ravel()
    df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
        .sort_values("Coefficient", ascending=False)
        .reset_index(drop=True)
    )
    top_pos = df.head(top_n)
    top_neg = df.tail(top_n)
    result = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values(
        "Coefficient", ascending=False
    )
    result = result.copy()
    result["Direction"] = result["Coefficient"].apply(
        lambda c: "Higher cancellation risk" if c > 0 else "Lower cancellation risk"
    )
    return result.reset_index(drop=True)
