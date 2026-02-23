from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_model

df = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

model = train_models(X_train, y_train)
evaluate_model(model, X_test, y_test)