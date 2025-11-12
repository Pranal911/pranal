import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


model_files = {
    "Logistic Regression": "ML_Model/Logistic_Regression_model.pkl",
    "Decision Tree Classifier": "ML_Model/Decision_Tree_Classifier_model.pkl",
    "Random Forest Classifier": "ML_Model/Random_Forest_Classifier_model.pkl",
    "KNN Classifier": "ML_Model/KNN_Classifier_model.pkl",
    "SVC": "ML_Model/SVC_model.pkl",
    "Naive Bayes ": "ML_Model/Naive_Bayes_model.pkl"
}

loaded_models = {}
for name, path in model_files.items():
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)


st.title(" Heart Disease Prediction System")
st.write(" Enter the technical details to check if your have heart disease or not")
st.warning(
    f"⚠️ Disclaimer: This system is developed by Pranal Thapa (student) using datasets from kaggle, and is for educational use only."
    " The results are based on algorithms and are NOT to be used for actual medical advice or diagnosis."
)
# User Inputs
age = st.number_input("Age", min_value=1, max_value=100,value=50)
sex= st.selectbox("Sex 0 if female 1 if male",[0,1])
cp= st.selectbox("Chest Pain Type",[0,1,2,3])
trestbps=st.number_input("Resting Blood Pressure",min_value=90,max_value=200,value=120)
chol= st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs= st.selectbox("Fasting Blood Sugar",[0,1])
restecg= st.selectbox("Resting Electrocardiographic Result",[0,1,2])
thalach = st.number_input("Max Heart rate achieved",min_value=70, max_value=220,value=150)
exang= st.selectbox("Exercise Induced Angina",[0,1])
oldpeak = st.number_input("ST Depression",min_value=0.0,max_value=6.2,value=1.0,step=0.1)
slope=st.selectbox("Slope of Exercise",[0,1,2])
ca=st.selectbox("Number of Major Vessels",[0,1,2,3])
thal=st.selectbox("Thalasemmia",[0,1,2,3])

# Convert to DataFrame
input_data = pd.DataFrame([[
    age,sex,cp,trestbps,chol,fbs,
    restecg,thalach,exang,oldpeak,slope,
    ca,thal
]], columns=[
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak","slope",
    "ca","thal"
])


st.subheader("Choose a Model for Prediction")

model_choice = st.selectbox("Select Model", list(loaded_models.keys()))

if st.button("Predict"):
    model = loaded_models[model_choice]
    prediction = model.predict(input_data)

    if prediction[0]==1:
        st.success("You may have a heart disease according to this calculation. Take Care- Pranal Thapa")
    else:
        st.success("You may not have a heart disease according to this calculation. Stay Safe and Healthy - Pranal Thapa")