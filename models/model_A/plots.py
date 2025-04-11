import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as m
import torch

# def plot_roc_curve(y_true, y_probs):
#     fpr, tpr, _ = m.roc_curve(y_true, y_probs)
#     roc_auc = m.auc(fpr, tpr)

#     fig, ax = plt.subplots(figsize=(6,6))
#     ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
#     ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
#     ax.set_xlabel('False Positive Rate')
#     ax.set_ylabel('True Positive Rate')
#     ax.set_title('ROC Curve')
#     ax.legend()
#     return fig


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = m.roc_curve(y_true, y_probs)
    roc_auc = m.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2, color='darkblue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

    ax.fill_between(fpr, tpr, alpha=0.1, color='blue')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('ROC Curve — Clinical Model')

    ax.text(0.6, 0.2, '↑ FP = More false alarms', fontsize=9, color='red')
    ax.text(0.1, 0.9, '↑ TP = More true detections', fontsize=9, color='green')

    ax.legend(loc='lower right')
    ax.grid(True)
    return fig


def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = m.precision_recall_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=["Negative", "Positive"]):
    cm = m.confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = f"True\n\n{c}\n({p:.1f}%)"
            else:
                s = f"False\n\n{c}\n({p:.1f}%)"
            annot[i, j] = s

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=annot, fmt='', cmap='Blues',cbar=False, linewidths=0.5, linecolor='black',annot_kws={"size": 16})

    plt.title('Confusion Matrix', fontsize=24)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    # Summary statistics
    accuracy = np.trace(cm) / float(np.sum(cm))
    precision = m.precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = m.recall_score(y_true, y_pred, average='binary', zero_division=0)
    f2 = m.fbeta_score(y_true, y_pred, beta=2, average='binary', zero_division=0)

    stats_text = f"""
    Accuracy = {accuracy:.2%}
    Precision = {precision:.2%}
    Recall = {recall:.2%}
    F2 Score = {f2:.2%}
    """
    plt.gcf().text(-0.3, 0, stats_text, fontsize=24, va="bottom")

    return fig

# TODO: add this plot


def plot_attention_heatmap(attention_weights, feature_names):

    avg_attention = np.mean(attention_weights, axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(avg_attention.reshape(1, -1), annot=True,
                cmap="viridis", xticklabels=feature_names, ax=ax)
    ax.set_title("Feature Importance (Attention Heatmap)")
    ax.set_yticks([])
    return fig


def plot_loss_per_epoch(loss_counter):
    """Generates and saves the loss per epoch plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(loss_counter) + 1), loss_counter, marker='o', linestyle='-')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Epoch")
    ax.grid(True)
    return fig


def get_feature_importance(model, feature_names):
    """
    Feature importance for Linear embedding: average absolute weight per input feature.
    """
    with torch.no_grad():
        # (d_model, input_dim) → (input_dim, d_model)
        emb_weights = model.embedding.weight.detach().T  
        emb_importance = torch.mean(torch.abs(emb_weights), dim=1)  # shape: (input_dim,)
        importance_scores = emb_importance.cpu().numpy()

    assert len(importance_scores) == len(feature_names), \
        f"Mismatch: {len(importance_scores)} scores vs {len(feature_names)} features"

    return dict(zip(feature_names, importance_scores))



def plot_top_10_features(model, feature_names):
    feature_importance = get_feature_importance(model, feature_names)

    # Sort features by importance
    sorted_features = dict(sorted(feature_importance.items(),
                                  key=lambda x: x[1],
                                  reverse=True))

    # Get top 10
    top_10 = dict(list(sorted_features.items())[:10])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(top_10.keys(), top_10.values())

    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    plt.tight_layout()
    return fig


def plot_less_10_features(model, feature_names):
    feature_importance = get_feature_importance(model, feature_names)

    # Sort features by importance
    sorted_features = dict(sorted(feature_importance.items(),
                                  key=lambda x: x[1],
                                  reverse=False))

    # Get bottom 10
    bottom_10 = dict(list(sorted_features.items())[:10])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(bottom_10.keys(), bottom_10.values())

    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.title('10 Least Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    plt.tight_layout()
    return fig


def save_plots(root, xperiment_name, loss_counter, y_test, y_probs, y_pred, model, feature_names):
    save_path = f"{root}/models/model_B/results/{xperiment_name}"
    os.makedirs(save_path, exist_ok=True)

    for plot_func, args, filename in [
        (plot_roc_curve, (y_test, y_probs), "roc_curve.png"),
        (plot_precision_recall_curve, (y_test, y_probs), "precision_recall_curve.png"),
        (plot_confusion_matrix, (y_test, y_pred), "confusion_matrix.png"),
        (plot_top_10_features, (model, feature_names), "10_important_features.png"),
        (plot_less_10_features, (model, feature_names), "10_less_important_features.png"),
        (plot_loss_per_epoch, (loss_counter,), "loss_per_epoch.png"),
    ]:
        fig = plot_func(*args)
        fig.savefig(os.path.join(save_path, filename), bbox_inches='tight')
        plt.close(fig)

def save_all_xgb_plots(y_true, y_pred, y_probs, save_dir,booster=None, feature_names=None, loss_per_epoch=None):
    """
    Save all evaluation-related plots for XGBoost:
    - ROC Curve
    - Precision-Recall Curve
    - Confusion Matrix
    - Feature Importance (requires booster and feature_names)
    - Top 10 Important Features
    - Loss per Epoch (if available)
    """
    os.makedirs(save_dir, exist_ok=True)

    # ROC Curve
    fig = plot_roc_curve(y_true, y_probs)
    fig.savefig(os.path.join(save_dir, "xgb_roc_curve.png"), bbox_inches='tight')
    plt.close(fig)

    # PR Curve
    fig = plot_precision_recall_curve(y_true, y_probs)
    fig.savefig(os.path.join(save_dir, "xgb_pr_curve.png"), bbox_inches='tight')
    plt.close(fig)

    # Confusion Matrix
    fig = plot_confusion_matrix(y_true, y_pred)
    fig.savefig(os.path.join(save_dir, "xgb_confusion_matrix.png"), bbox_inches='tight')
    plt.close(fig)


    # Loss Curve
    if loss_per_epoch is not None:
        fig = plot_loss_per_epoch(loss_per_epoch)
        fig.savefig(os.path.join(save_dir, "xgb_loss_curve.png"), bbox_inches='tight')
        plt.close(fig)
