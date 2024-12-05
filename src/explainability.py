import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from pathlib import Path
import logging


def analyze_stacking_model(model, X_test, y_test, y_pred, y_pred_proba, output_dir):
    """
    Analyze stacking model performance and generate visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate feature importance
    importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': importance.importances_mean
    }).sort_values('importance', ascending=False)

    # Create and save feature importance plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()

    # Create and save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

    # Calculate metrics
    results = {
        'feature_importance': importance_df.to_dict('records'),
        'confidence': {
            'mean': float(np.mean(np.max(y_pred_proba, axis=1))),
            'correct_pred': float(np.mean(np.max(y_pred_proba[y_pred == y_test], axis=1)))
        },
        'errors': {
            'count': int((y_pred != y_test).sum()),
            'rate': float((y_pred != y_test).mean())
        }
    }

    # Save text report
    report = [
        "# Model Analysis Report",
        f"\nMean confidence: {results['confidence']['mean']:.4f}",
        f"Confidence on correct predictions: {results['confidence']['correct_pred']:.4f}",
        f"Number of errors: {results['errors']['count']}",
        f"Error rate: {results['errors']['rate']:.4f}",
        "\nTop 10 most important features:"
    ]

    for feat in importance_df.head(10).itertuples():
        report.append(f"- {feat.feature}: {feat.importance:.4f}")

    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))

    return results
