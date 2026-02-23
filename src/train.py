import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.config import MODEL_PATH
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_models(X_train, y_train):

    models = {
        "logistic": LogisticRegression(max_iter=5000),
        "random_forest": RandomForestClassifier()
    }
    
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(eval_metric="logloss")

    param_grid = {
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 10]
        }
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        if name in param_grid:
            grid = GridSearchCV(model, param_grid[name], cv=5, scoring="roc_auc")
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)

        score = model.score(X_train, y_train)

        if score > best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, MODEL_PATH)
    print("Best model saved.")

    return best_model