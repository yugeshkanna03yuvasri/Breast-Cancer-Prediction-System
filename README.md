#  Breast Cancer Prediction System

An end-to-end Machine Learning system for breast cancer diagnosis using tumor characteristics.

 Achieves **97% Accuracy** with **0.994 ROC-AUC Score**

---

##  Project Overview

This project builds a complete machine learning pipeline to classify breast tumors as **Malignant (1)** or **Benign (0)** using diagnostic measurements.

It includes:
- Model training & evaluation
- Hyperparameter tuning
- Model comparison
- Performance visualization
- REST API
- Interactive web interface

---

##  Features

- ✅ Multiple ML Algorithms (Logistic Regression, Random Forest, SVM)
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Interactive Web Interface (Streamlit)
- ✅ REST API using FastAPI
- ✅ Model performance visualization dashboard
- ✅ Structured project architecture
- ✅ Comprehensive logging system
- ✅ Model comparison reports

---

##  Model Performance

| Metric      | Score        |
|------------|-------------|
| Accuracy   | 96%         |
| Precision  | 95% – 100%  |
| Recall     | 90% – 100%  |
| ROC-AUC    | 0.994       |

 The high ROC-AUC score indicates excellent class separability.

---

##  Quick Start

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Train the Model

```bash
python main.py
```

###  Run the Streamlit Web App

```bash
streamlit run app_streamlit.py
```

###  Run the FastAPI Server

```bash
uvicorn api.app:app --reload
```

---

##  Project Structure

```
Breast-Cancer-Prediction-Project/
│
├── src/
│   ├── config.py            # Configuration settings
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py     # Data preprocessing
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation
│   ├── predict.py           # Prediction utilities
│   └── logger.py            # Logging system
│
├── api/
│   └── app.py               # FastAPI application
│
├── data/raw/                # Dataset storage
├── models/                  # Trained models
├── logs/                    # Application logs
│
├── main.py                  # Main training script
├── app_streamlit.py         # Streamlit web app
├── compare_models.py        # Model comparison script
├── visualize_results.py     # Performance visualization
└── test_predictions.py      # Prediction testing script
```

---

##  Usage Examples

### ▶ Command Line Prediction

```bash
python test_predictions.py
```

###  Compare Models

```bash
python compare_models.py
```

### 📈 Generate Performance Visualizations

```bash
python visualize_results.py
```

---

##  API Endpoint

### POST `/predict`

#### Request Body:

```json
{
  "features": [14.13, 19.29, 91.97, 578.3, 0.113, ...]
}
```

#### Response:

```json
{
  "prediction": 0,
  "probability": 0.95
}
```

- `prediction = 0` → Benign  
- `prediction = 1` → Malignant  

---

##  Dataset

**UCI Breast Cancer Wisconsin (Diagnostic) Dataset**

- 🔢 Total Samples: 569
- 📊 Features: 30 numeric tumor features
- 🎯 Target Classes:
  - 1 → Malignant
  - 0 → Benign

---

##  Technologies Used

- Python 3.11
- scikit-learn
- FastAPI
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

##  Future Improvements

- 🔹 Model deployment on cloud (AWS / GCP / Azure)
- 🔹 Docker containerization
- 🔹 CI/CD integration
- 🔹 Deep Learning experimentation
- 🔹 Explainable AI (SHAP / LIME)

---

##  License

MIT License

---

##  Author

**Yuvasri K**  

---

