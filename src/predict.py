import joblib
from src.config import MODEL_PATH

def predict(input_data):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([input_data])
    return prediction[0]