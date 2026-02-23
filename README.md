# 🏥 Breast Cancer Prediction System

An end-to-end machine learning system for breast cancer diagnosis using tumor characteristics. Achieves **96% accuracy** with **0.994 ROC-AUC score**.

## 🎯 Features

- ✅ Multiple ML algorithms (Logistic Regression, Random Forest, SVM)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Interactive web interface (Streamlit)
- ✅ REST API (FastAPI)
- ✅ Model performance visualization
- ✅ Comprehensive logging system
- ✅ Model comparison reports

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 96% |
| Precision | 95-100% |
| Recall | 90-100% |
| ROC-AUC | 0.994 |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python main.py
```

### 3. Run Web Interface
```bash
streamlit run app_streamlit.py
```

### 4. Run API Server
```bash
uvicorn api.app:app --reload
```

## 📁 Project Structure
```
Breast-Cancer-Prediction Project/
├── src/
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   ├── predict.py          # Prediction utilities
│   └── logger.py           # Logging system
├── api/
│   └── app.py              # FastAPI application
├── data/raw/               # Dataset storage
├── models/                 # Trained models
├── logs/                   # Application logs
├── main.py                 # Main training script
├── app_streamlit.py        # Streamlit web app
├── compare_models.py       # Model comparison
├── visualize_results.py    # Performance visualization
└── test_predictions.py     # Testing script
```

## 🔬 Usage Examples

### Command Line Prediction
```bash
python test_predictions.py
```

### Model Comparison
```bash
python compare_models.py
```

### Generate Visualizations
```bash
python visualize_results.py
```

## 🌐 API Endpoints

**POST** `/predict`
```json
{
  "features": [14.13, 19.29, 91.97, ...]
}
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.95
}
```

## 📈 Dataset

UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- **Samples**: 569
- **Features**: 30 numeric features
- **Classes**: Malignant (1), Benign (0)

## 🛠️ Technologies

- Python 3.11
- scikit-learn
- FastAPI
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn

## 📝 License

MIT License

## 👤 Author

Your Name - Breast Cancer Prediction System

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

⚠️ **Disclaimer**: This is a diagnostic support tool. Always consult healthcare professionals for medical decisions.
