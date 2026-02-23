import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_loader import load_data
from src.preprocessing import preprocess_data

df = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

df_results = pd.DataFrame(results)
df_results = df_results.round(4)

print("\n" + "="*70)
print("MODEL COMPARISON REPORT".center(70))
print("="*70 + "\n")
print(df_results.to_string(index=False))
print("\n" + "="*70)

# Save to CSV
df_results.to_csv('models/model_comparison.csv', index=False)
print("\n✓ Report saved to models/model_comparison.csv")
