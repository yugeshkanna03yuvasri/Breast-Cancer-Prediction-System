import pandas as pd
import os
from src.config import DATA_PATH
from sklearn.datasets import load_breast_cancer

def load_data():
    if os.path.exists(DATA_PATH):
        print("Loading dataset from UCI CSV...")
        df = pd.read_csv(DATA_PATH, header=None)
       
        diagnosis = df.iloc[:, 1].map({'M': 1, 'B': 0})
        features = df.iloc[:, 2:]
       
        df = features.copy()
        df.columns = [f'feature_{i}' for i in range(len(df.columns))]
        df['label'] = diagnosis.values
        
        return df
    else:
        print("Dataset not found. Creating from sklearn...")
        dataset = load_breast_cancer()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df["label"] = dataset.target
        return df