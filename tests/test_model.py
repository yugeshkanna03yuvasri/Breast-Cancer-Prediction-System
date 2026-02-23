import unittest
import numpy as np
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
import joblib
import os

class TestBreastCancerPrediction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = 'models/best_model.pkl'
        
    def test_data_loading(self):
        """Test if data loads correctly"""
        df = load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('label', df.columns)
        self.assertGreater(len(df), 0)
    
    def test_data_preprocessing(self):
        """Test data preprocessing"""
        df = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        self.assertEqual(len(X_train.shape), 2)
        self.assertEqual(len(X_test.shape), 2)
        self.assertGreater(len(X_train), len(X_test))
    
    def test_model_exists(self):
        """Test if trained model exists"""
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_model_prediction(self):
        """Test model prediction"""
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            sample = np.random.rand(1, 30)
            prediction = model.predict(sample)
            self.assertIn(prediction[0], [0, 1])
    
    def test_model_probability(self):
        """Test model probability output"""
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            sample = np.random.rand(1, 30)
            proba = model.predict_proba(sample)
            self.assertEqual(proba.shape, (1, 2))
            self.assertAlmostEqual(np.sum(proba), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
