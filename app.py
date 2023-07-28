import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from viz.EDA import gender_stroke_plot

data=pd.read_csv('data/brain_stroke.csv')

scaler = joblib.load("scaler.joblib")
model = joblib.load("decision_tree.joblib")

def main():
    st.title("Brain Stroke Prediction")
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["Introduction", "Gender vs Stroke","Model Prediction"])

    if page=="Introduction":
        st.header("Background")
        st.write("Relationship of Gender and Stroke")
        

    elif page=="Gender vs Stroke":
        st.header("Background")
        st.write("Relationship of Gender and Stroke")

        st.pyplot(gender_stroke_plot(data))

    elif page=="Model Prediction":
        age = st.text_input("AGE", "0.0")
        hypertension = st.text_input("HYPERTENSION", "0.0")
        heart_disease = st.text_input("HEART DISEASE", "0.0")
        avg_glucose_level = st.text_input("AVERAGE GLUCOSE LEVEL", "0.0")
        bmi = st.text_input("BODY MASS INDEX (BMI)", "0.0")
        smoking_status = st.text_input("SMOKING STATUS", "0.0")

        inputs = [float(age), float(hypertension), float(heart_disease), float(avg_glucose_level), float(bmi), float(smoking_status)]
        if st.button("RUN"):
            input_data = np.array(inputs).reshape(1,-1)
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)
            st.write(f"Prediction: {prediction[0]}")

if __name__== "__main__":
    main()

