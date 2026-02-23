import joblib
import pandas as pd
from src.data_loader import load_data

# Load the trained model
model = joblib.load('models/best_model.pkl')

# Load data to get sample features
df = load_data()
X = df.drop('label', axis=1)
y = df['label']

# Test on first 5 samples
print("Testing predictions on 5 samples:\n")
for i in range(5):
    sample = X.iloc[i:i+1].values
    prediction = model.predict(sample)[0]
    actual = y.iloc[i]
    probability = model.predict_proba(sample)[0]
    
    result = "Malignant" if prediction == 1 else "Benign"
    actual_result = "Malignant" if actual == 1 else "Benign"
    
    print(f"Sample {i+1}:")
    print(f"  Predicted: {result} (confidence: {max(probability):.2%})")
    print(f"  Actual: {actual_result}")
    print(f"  Correct: {'✓' if prediction == actual else '✗'}\n")
