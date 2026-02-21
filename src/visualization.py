"""
Visualization utilities for cardiovascular risk prediction.

This module centralizes all figure generation for evaluation,
calibration, interpretability, and model diagnostics. It does not
perform any computation; it only receives precomputed data and
returns matplotlib figures.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import confusion_matrix

# Clinical visualization limits applied only to improve readability of blood pressure histograms.
# These limits do not modify the dataset and are not part of data cleaning or preprocessing.
def plot_histograms(df: pd.DataFrame, cols: list):
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 3.5 * len(cols)))

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):

        # Clinical visualization limits (not cleaning)
        if col == "ap_hi":
            data = df[col].clip(lower=70, upper=250)
        elif col == "ap_lo":
            data = df[col].clip(lower=40, upper=150)
        else:
            data = df[col]

        sns.histplot(data, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)

    fig.tight_layout()
    return fig

def plot_categorical_bars(df: pd.DataFrame, cols: list, labels: dict):
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 3.5 * len(cols)))

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):

        # Category mapping
        mapping = labels.get(col)
        categories = list(mapping.keys())

        # Count values and include missing categories
        counts = df[col].value_counts().reindex(categories, fill_value=0)

        # One color per category
        palette = sns.color_palette("pastel", len(categories))

        # Draw bars
        bars = ax.bar(categories, counts.values, color=palette)

        # Titles and labels
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        # Legend: one label per bar
        legend_labels = [mapping[c] for c in categories]
        ax.legend(bars, legend_labels, title=col)

    fig.tight_layout()
    return fig

def plot_numeric_vs_target(df: pd.DataFrame, cols: list, target: str = "cardio"):
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 3.5 * len(cols)))

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.boxplot(x=df[target], y=df[col], ax=ax)
        ax.set_title(f"{col} by {target}")
        ax.set_xlabel(target)
        ax.set_ylabel(col)

    fig.tight_layout()
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,      
        fmt=".2f",       
        ax=ax
    )
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig, corr

def plot_categorical_vs_target(df: pd.DataFrame, cols: list, target: str = "cardio"):
    tabla = pd.DataFrame()

    for col in cols:
        cruzada = pd.crosstab(df[col], df[target], normalize="index") * 100
        cruzada["variable"] = col
        cruzada["categoria"] = cruzada.index
        tabla = pd.concat([tabla, cruzada])

    tabla_pivot = tabla.pivot(index="variable", columns="categoria", values=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(tabla_pivot, annot=True, fmt=".1f", cmap="Blues", ax=ax)
    ax.set_title("Percentage of cardio=1 by category")
    fig.tight_layout()
    return fig, tabla_pivot

def plot_roc_curve(fpr, tpr, auc_value, model_name):
    """
    Generate ROC curve figure with model name and AUC value.

    Parameters
    ----------
    fpr : array-like
        False positive rates.
    tpr : array-like
        True positive rates.
    auc_value : float
        ROC-AUC score to display.
    model_name : str
        Name of the model.

    Returns
    -------
    matplotlib.figure.Figure
        ROC curve figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")

    ax.text(
        0.60, 0.05,
        f"AUC = {auc_value:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    fig.tight_layout()
    return fig

def plot_pr_curve(precision, recall, pr_value, model_name):
    """
    Generate Precision–Recall curve with model name and PR-AUC value.

    Parameters
    ----------
    precision : array-like
        Precision values.
    recall : array-like
        Recall values.
    pr_value : float
        PR-AUC score to display.
    model_name : str
        Name of the model.

    Returns
    -------
    matplotlib.figure.Figure
        PR curve figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve – {model_name}")

    ax.text(
        0.60, 0.05,
        f"PR-AUC = {pr_value:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    fig.tight_layout()
    return fig

def plot_calibration_summary(models_data, bins):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for model_name, conf, acc in models_data:
        ax1.plot(conf, acc, marker="o", label=model_name)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Reliability Curve – All Models")
    ax1.legend()

    ax2 = axes[1]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 0.25
    offsets = [-width, 0, width]

    for (model_name, conf, acc), offset in zip(models_data, offsets):
        ece_values = np.abs(conf - acc)
        ax2.bar(bin_centers + offset, ece_values, width=width, label=model_name)

    ax2.set_xlabel("Probability bin")
    ax2.set_ylabel("ECE per bin")
    ax2.set_title("ECE per Bin – All Models")
    ax2.legend()

    fig.tight_layout()
    return fig

def plot_confusion_matrix_threshold(y_true, y_prob, threshold, title="Confusion Matrix"):
    """
    Plot a confusion matrix at a given threshold.
    """
    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax 
    ) 
    ax.set_title(f"{title} (thr={threshold:.4f})") 
    ax.set_xlabel("Predicted") 
    ax.set_ylabel("Actual") 
    fig.tight_layout() 
    
    return fig

def plot_metrics_barplot(results_dict, title="Performance Metrics"):
    """
    Plot sensitivity, specificity, PPV, NPV, F1 as a barplot.
    """
    metrics = ["sensitivity", "specificity", "PPV", "NPV", "F1"]
    values = [results_dict[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=metrics, y=values, color="steelblue", ax=ax) 
    ax.set_ylim(0, 1) 
    ax.set_title(title) 
    ax.set_ylabel("Score") 
    fig.tight_layout() 
    
    return fig

def plot_permutation_importance(pi_df, title="Permutation Importance"):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(
        pi_df["feature"],
        pi_df["importance_mean"],
        xerr=pi_df["importance_std"],
        color="steelblue",
        alpha=0.8
    )

    ax.set_xlabel("Mean Importance (Δ ROC-AUC)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.invert_yaxis()  # Most important at the top
    fig.tight_layout()

    return fig

def plot_pdp(grid, values, feature_name):
    """
    Generate Partial Dependence plot.

    Parameters
    ----------
    grid : array-like
        Feature grid values.
    values : array-like
        PDP values.
    feature_name : str
        Name of the feature.

    Returns
    -------
    matplotlib.figure.Figure
        PDP figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, values)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Partial Dependence")
    ax.set_title(f"PDP – {feature_name}")
    fig.tight_layout()
    return fig

def plot_shap_summary(shap_values, feature_names):
    """
    SHAP summary plot (global importance).
    """
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    return plt.gcf()

def plot_shap_dependence(shap_values, X, pipeline, feature):
    """
    SHAP dependence plot using the ORIGINAL feature name.
    The function automatically maps the original feature name
    to the transformed feature name inside the ColumnTransformer.
    """
    preprocessor = pipeline.named_steps["prep"]
    feature_names = preprocessor.get_feature_names_out()

    matches = [f for f in feature_names if feature in f]
    if len(matches) == 0:
        raise ValueError(f"Feature '{feature}' not found in transformed feature names.")
    if len(matches) > 1:
        raise ValueError(f"Feature '{feature}' matches multiple transformed features: {matches}")

    transformed_feature = matches[0]

    idx = list(feature_names).index(transformed_feature)

    X_trans = preprocessor.transform(X)

    shap.dependence_plot(
        idx,
        shap_values,
        X_trans,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    return plt.gcf()

def plot_shap_interaction_pair(interaction_values, X, pipeline, feature1, feature2):
    """
    SHAP interaction plot for two original feature names.
    Automatically maps original names to the correct transformed numeric features.
    """
    preprocessor = pipeline.named_steps["prep"]
    feature_names = preprocessor.get_feature_names_out()

    def find_transformed(original_name):
        matches = [f for f in feature_names if original_name in f]

        if len(matches) == 0:
            raise ValueError(f"Feature '{original_name}' not found in transformed names.")

        numeric_matches = [f for f in matches if f.startswith("num__")]
        if len(numeric_matches) == 1:
            return numeric_matches[0]

        raise ValueError(
            f"Feature '{original_name}' matches multiple transformed names: {matches} "
            f"and no unique numeric feature was found."
        )

    f1 = find_transformed(feature1)
    f2 = find_transformed(feature2)

    idx1 = list(feature_names).index(f1)
    idx2 = list(feature_names).index(f2)

    X_trans = preprocessor.transform(X)

    shap.dependence_plot(
        (idx1, idx2),
        interaction_values,
        X_trans,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    return plt.gcf()
