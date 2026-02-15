# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 19:42:07 2026

@author: kuchhal
"""

import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report,ConfusionMatrixDisplay
)

st.set_page_config(page_title="Forest Cover Type Classifier", layout="wide")

st.title("🌲 Forest Cover Type Classification")

st.write("This app predicts forest cover type using ML models.")

# Sidebar
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Load model (you must save these models earlier)
model_files = {
    "Logistic Regression": "model/logistic_regression_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "XGBoost": "model/xgboost_model.pkl"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]
    

    model = joblib.load(model_files[model_choice])
    scaler = joblib.load("model/scaler.pkl")
    X_scaled = scaler.transform(X)

    if model_choice in ["Logistic Regression", "KNN", "XGBoost"]:
        y_pred = model.predict(X_scaled)
    else:
        y_pred = model.predict(X)

    
    if model_choice == "XGBoost":
        y_pred=y_pred+1
    st.write("### Model Performance")
    st.write(f"### Selected Model: {model_choice}")

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted")
    rec = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)
    #auc = roc_auc_score(y, y_prob, multi_class="ovr")

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 score:", f1)
    st.write("mcc:", mcc)
    #st.write("Precision:", prec)
    st.text("Classification Report:")
    st.text(classification_report(y, y_pred))
    
    cm = confusion_matrix(y, y_pred)
        
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
