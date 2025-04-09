import streamlit as st
import pandas as pd
import joblib
import os


current_dir = os.getcwd()
model_path = os.path.join(current_dir, "outputs", "models", "best_model.pkl")
data_path = os.path.join(current_dir, "outputs", "datasets", "cleaned", "TrainSetCleaned.csv")


st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("House Price Prediction App")


if not os.path.exists(model_path):
    st.error(f"Model couldn't be found. Trying to load from: {model_path}")
    st.stop()


try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.subheader("Example of training data:")
    st.dataframe(df.head())
else:
    st.error(f"Data file could not be found: {data_path}")
    st.stop()


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
        st.success(f"Predicted price: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"Something went wrong in the prediction: {e}")
