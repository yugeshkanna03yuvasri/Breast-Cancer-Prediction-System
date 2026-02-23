import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42