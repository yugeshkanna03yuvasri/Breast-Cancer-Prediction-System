import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.data_loader import load_data
from src.preprocessing import preprocess_data

print("Loading model and data...")
model = joblib.load('models/best_model.pkl')
df = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

print("Making predictions...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ✅ Use constrained_layout to prevent overlapping
fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)

fig.suptitle(
    'Breast Cancer Prediction Model Performance',
    fontsize=20,
    fontweight='bold'
)

# 1️⃣ Confusion Matrix
print("\n📊 Chart 1: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0, 0],
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant'],
    cbar_kws={'label': 'Count'}
)

axes[0, 0].set_title(
    'Confusion Matrix\n(Actual vs Predicted)',
    fontsize=14,
    pad=15
)
axes[0, 0].set_ylabel('Actual Label', fontsize=12)
axes[0, 0].set_xlabel('Predicted Label', fontsize=12)

textstr = f'Total: {len(y_test)} samples\nCorrect: {cm[0,0] + cm[1,1]}\nWrong: {cm[0,1] + cm[1,0]}'

axes[0, 0].text(
    0.02,
    0.98,
    textstr,
    transform=axes[0, 0].transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

# 2️⃣ ROC Curve
print("\n📈 Chart 2: ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

axes[0, 1].plot(
    fpr,
    tpr,
    color='darkorange',
    lw=3,
    label=f'Model (AUC = {roc_auc:.4f})'
)

axes[0, 1].plot(
    [0, 1],
    [0, 1],
    color='navy',
    lw=2,
    linestyle='--',
    label='Random Guess (AUC = 0.5)'
)

axes[0, 1].fill_between(fpr, tpr, alpha=0.2, color='darkorange')

axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)

axes[0, 1].set_title(
    'ROC Curve\n(Higher = Better)',
    fontsize=14,
    pad=15
)

axes[0, 1].legend(loc="lower right", fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3️⃣ Feature Importance
print("\n⭐ Chart 3: Feature Importance")

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 10))

    axes[1, 0].barh(range(10), importances[indices], color=colors)
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels(
        [f'Feature {i}' for i in indices],
        fontsize=10
    )

    axes[1, 0].set_xlabel('Importance Score', fontsize=12)

    axes[1, 0].set_title(
        'Top 10 Most Important Features\n(Higher = More Important)',
        fontsize=14,
        pad=15
    )

    axes[1, 0].grid(True, alpha=0.3, axis='x')

else:
    axes[1, 0].text(
        0.5,
        0.5,
        'Feature importance\nnot available\nfor this model',
        ha='center',
        va='center',
        fontsize=14
    )
    axes[1, 0].set_title('Feature Importance', fontsize=14, pad=15)

# 4️⃣ Prediction Probability Distribution
print("\n📊 Chart 4: Prediction Probability Distribution")

benign_probs = y_proba[y_test == 0]
malignant_probs = y_proba[y_test == 1]

axes[1, 1].hist(
    benign_probs,
    bins=20,
    label=f'Benign (n={len(benign_probs)})',
    color='green',
    alpha=0.6,
    edgecolor='black'
)

axes[1, 1].hist(
    malignant_probs,
    bins=20,
    label=f'Malignant (n={len(malignant_probs)})',
    color='red',
    alpha=0.6,
    edgecolor='black'
)

axes[1, 1].axvline(
    x=0.5,
    color='black',
    linestyle='--',
    linewidth=2,
    label='Decision Threshold'
)

axes[1, 1].set_xlabel(
    'Predicted Probability of Malignancy',
    fontsize=12
)

axes[1, 1].set_ylabel(
    'Number of Samples',
    fontsize=12
)

axes[1, 1].set_title(
    'Prediction Confidence Distribution\n(Left=Benign, Right=Malignant)',
    fontsize=14,
    pad=15
)

axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Save figure
plt.savefig(
    'models/model_performance.png',
    dpi=300,
    bbox_inches='tight'
)

print("\n✓ Visualization saved to models/model_performance.png")

print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)
print(f"Total Test Samples: {len(y_test)}")
print(f"Correct Predictions: {(y_pred == y_test).sum()}")
print(f"Accuracy: {(y_pred == y_test).sum() / len(y_test) * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("="*70)

plt.show()
