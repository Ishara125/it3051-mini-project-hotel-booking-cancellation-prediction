"""Reusable helper functions for Member 4 - Gradient Boosting Classifier.

These utilities handle evaluation, visualization, and reporting so the
notebook stays clean and readable for undergraduate students.

Member 4 responsibilities (Evaluation 2):
  - Gradient Boosting baseline and tuned model
  - Evaluation metrics, confusion matrix, ROC curve
  - Overfitting investigation (training vs test performance)
  - Hyperparameter tuning with cross-validation on training data only
  - Feature importance and ensemble interpretation
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

# -- Colour palette for Member 4 plots ---------------------------------------
_PALETTE = {
    "primary":   "#4A148C",  # deep purple - Gradient Boosting theme
    "secondary": "#7B1FA2",  # vibrant purple
    "positive":  "#D84315",  # burnt orange - cancelled class
    "negative":  "#1565C0",  # blue         - not-cancelled class
    "light_bg":  "#F9F9FB",
    "grid":      "#E2E2E6",
}


# -- 1. Comprehensive evaluation --------------------------------------------

def evaluate_classifier(
    y_true,
    y_pred,
    y_prob,
    label: str = "Model",
) -> dict:
    """Calculate and print the full set of classification metrics.

    Parameters
    ----------
    y_true : array-like  True binary labels (0=Not cancelled, 1=Cancelled).
    y_pred : array-like  Predicted class labels.
    y_prob : array-like  Predicted probabilities for class 1 (Cancelled).
    label  : str         Display name for the printed header.

    Returns
    -------
    dict  Metric name -> float, ready for a comparison table.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    border = "=" * 62
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
    print(classification_report(
        y_true, y_pred,
        target_names=["Not Cancelled", "Cancelled"]
    ))

    return {
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1-score":  round(f1,   4),
        "ROC-AUC":   round(auc,  4),
    }


# -- 2. Confusion matrix plot ------------------------------------------------

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
    annot = np.array([
        [f"{tn:,}\nTrue Negative\n(Correct)", f"{fp:,}\nFalse Positive\n(False alarm)"],
        [f"{fn:,}\nFalse Negative\n(Missed)",  f"{tp:,}\nTrue Positive\n(Caught)"],
    ])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Purples",
        linewidths=1.5,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.show()
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


# -- 3. ROC curve plot -------------------------------------------------------

def plot_roc_curve(
    y_true,
    y_prob,
    label: str = "Gradient Boosting",
    figsize: tuple = (7, 5),
) -> None:
    """Plot the ROC curve and shade the area under it.

    The diagonal dashed line represents a random classifier (AUC=0.50).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_PALETTE["light_bg"])
    ax.plot(fpr, tpr, color=_PALETTE["primary"], lw=2.5,
            label=f"{label}  (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=_PALETTE["primary"])
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="Random classifier (AUC = 0.50)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)", fontsize=11)
    ax.set_title("ROC Curve - Gradient Boosting", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.8)
    plt.tight_layout()
    plt.show()
    print(f"  ROC-AUC = {auc:.4f}")


# -- 4. Overfitting check ----------------------------------------------------

def overfitting_check(
    model,
    X_train_proc,
    y_train,
    X_test_proc,
    y_test,
) -> pd.DataFrame:
    """Compare training vs test accuracy and F1 to investigate overfitting."""
    y_train_pred = model.predict(X_train_proc)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1  = f1_score(y_train, y_train_pred, zero_division=0)

    y_test_pred = model.predict(X_test_proc)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred, zero_division=0)

    df = pd.DataFrame({
        "Split":    ["Training", "Test", "Gap (Train - Test)"],
        "Accuracy": [f"{train_acc:.4f}", f"{test_acc:.4f}",
                     f"{train_acc - test_acc:.4f}"],
        "F1-score": [f"{train_f1:.4f}",  f"{test_f1:.4f}",
                     f"{train_f1 - test_f1:.4f}"],
    }).set_index("Split")
    return df


# -- 5. Baseline-vs-tuned comparison table -----------------------------------

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


# -- 6. Feature importance bar chart ----------------------------------------

def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    figsize: tuple = (9, 7),
) -> pd.DataFrame:
    """Extract and plot the top-N most important features.

    Gradient Boosting computes feature importance as the normalised total
    reduction in loss function (deviance) brought by each feature across all trees.

    IMPORTANT: These values show how much the model relied on each feature.
    They do NOT prove that a feature causes cancellations.

    Returns
    -------
    pd.DataFrame  Feature name, importance value, cumulative importance.
    """
    importances = model.feature_importances_
    df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df["Cumulative"] = df["Importance"].cumsum().round(4)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        df["Feature"][::-1],
        df["Importance"][::-1],
        color=_PALETTE["primary"],
        edgecolor="white",
        alpha=0.88,
    )
    ax.set_title(
        f"Gradient Boosting - Top {top_n} Feature Importances\n"
        "(Higher = model relied more heavily on this feature)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Importance (Normalised Impurity / Deviance Reduction)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_facecolor(_PALETTE["light_bg"])
    ax.grid(axis="x", color=_PALETTE["grid"], linewidth=0.8)
    plt.tight_layout()
    plt.show()

    return df
