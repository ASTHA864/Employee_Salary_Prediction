import streamlit as st
import joblib
import numpy as np

st.title("Employee_Salary_ Prediction")

st.divider()

st.write("With this app, you can predict the salary of an employee based on their years of experience")


years= st.number_input("Years of Experience", value=1, step=1, min_value=0)  
jobrate = st.number_input("Job Rate", value=3.5, step=0.5, min_value=0.0)  

x=[years,jobrate]
model = joblib.load("salary_prediction_model.pkl")
st.divider()
predict = st.button("Predict Salary")
st.divider()

if predict:
    st.balloons()
    x_input = np.array(x).reshape(1, -1)
    prediction = model.predict(x_input)[0]
    st.write(f"The predicted salary is: {prediction:.2f}")
else:
    st.write("Click the button to predict the salary.")