import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("outputs/models/best_model.pkl")  # ändra vid behov
data = pd.read_csv("outputs/datasets/cleaned/TrainSetCleaned.csv")

st.title("Heritage Housing Price Predictor")
st.write("Explore house attributes and predict sale prices in Ames, Iowa.")

st.sidebar.header(" Enter house features")

def user_input():
    overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.sidebar.slider("Ground Living Area (sq ft)", 334, 5642, 1500)
    garage_area = st.sidebar.slider("Garage Area (sq ft)", 0, 1418, 400)
    total_bsmt_sf = st.sidebar.slider("Total Basement (sq ft)", 0, 6110, 800)
    year_built = st.sidebar.slider("Year Built", 1872, 2010, 1970)

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageArea": [garage_area],
        "TotalBsmtSF": [total_bsmt_sf],
        "YearBuilt": [year_built]
    })
    return input_data

input_df = user_input()

if st.button("Predict Sale Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Sale Price: ${int(prediction[0]):,}")

st.header("Correlation Heatmap")

corr = data.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr[["SalePrice"]].sort_values(by="SalePrice", ascending=False), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
