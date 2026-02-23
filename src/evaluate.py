from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, predictions))

    print("ROC AUC:", roc_auc_score(y_test, proba))