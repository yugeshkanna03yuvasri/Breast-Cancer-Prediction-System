import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🏥", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

model = load_model()

# Title
st.title("🏥 Breast Cancer Prediction System")
st.markdown("### AI-Powered Diagnostic Assistant")

# Sidebar
st.sidebar.header("Patient Information")
st.sidebar.info("Enter the tumor characteristics below")

# Create input fields for 30 features
st.sidebar.subheader("Tumor Measurements")

# Sample default values (mean values)
defaults = [14.13, 19.29, 91.97, 654.89, 0.096, 0.104, 0.089, 0.048, 0.181, 0.063,
            0.406, 1.217, 2.866, 40.34, 0.007, 0.025, 0.032, 0.012, 0.021, 0.004,
            16.27, 25.68, 107.26, 880.58, 0.132, 0.254, 0.272, 0.114, 0.290, 0.084]

features = []
col1, col2 = st.sidebar.columns(2)

for i in range(30):
    with col1 if i < 15 else col2:
        val = st.number_input(f'Feature {i+1}', value=float(defaults[i]), format="%.4f", key=f'feat_{i}')
        features.append(val)

# Predict button
if st.sidebar.button("🔍 Predict", type="primary"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Diagnosis", "Malignant" if prediction == 1 else "Benign")
    
    with col2:
        st.metric("Confidence", f"{max(probability)*100:.2f}%")
    
    with col3:
        risk = "High" if prediction == 1 else "Low"
        st.metric("Risk Level", risk)
    
    # Visual indicator
    if prediction == 1:
        st.error("⚠️ Malignant tumor detected. Please consult with an oncologist immediately.")
    else:
        st.success("✅ Benign tumor detected. Regular monitoring recommended.")
    
    # Probability visualization
    st.subheader("Prediction Probabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        prob_df = pd.DataFrame({
            'Class': ['Benign', 'Malignant'],
            'Probability': [probability[0]*100, probability[1]*100]
        })
        st.bar_chart(prob_df.set_index('Class'))
    
    with col2:
        # Pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#90EE90', '#FF6B6B']
        ax.pie([probability[0], probability[1]], 
               labels=['Benign', 'Malignant'],
               autopct='%1.1f%%',
               colors=colors,
               startangle=90)
        ax.set_title('Prediction Distribution')
        st.pyplot(fig)

# Information section
st.markdown("---")
st.subheader("ℹ️ About This System")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Model Performance:**
    - Accuracy: 96%
    - ROC-AUC Score: 0.994
    - Model: Random Forest Classifier
    """)

with col2:
    st.write("""
    **Dataset Information:**
    - Samples: 569 cases
    - Features: 30 tumor characteristics
    - Classes: Malignant & Benign
    """)

st.warning("⚠️ Disclaimer: This is a diagnostic support tool. Always consult healthcare professionals for medical decisions.")
