import streamlit as st
import pandas as pd
import joblib
import os

model_path = "./outputs/models/best_model.pkl"
data_path = "./outputs/datasets/cleaned/TrainSetCleaned.csv"

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title(" House Price Prediction App")

if not os.path.exists(model_path):
    st.error("Model could't be found try another way.")
    st.stop()

model = joblib.load(model_path)

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.subheader("Exampel of training the data:")
    st.dataframe(df.head())


st.subheader("Make your own prediction:")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input("Above ground living area (GrLivArea)", min_value=500, max_value=5000, value=1500)
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)

with col2:
    garage_cars = st.slider("Garage cars", 0, 5, 2)
    total_bsmt_sf = st.number_input("Basement square feet (TotalBsmtSF)", min_value=0, max_value=3000, value=800)


if st.button("Predict price"):
    input_df = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "OverallQual": overall_qual,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predict price: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"Something went wrong In the pridiction: {e}")
