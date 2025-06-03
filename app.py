import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


st.title("🏡 House Price Predictor")
st.write("Use this app to predict house prices based on selected property features.")


st.sidebar.title("About the App")
st.sidebar.info(
    """
    This application uses a machine learning model to predict house prices 
    based on property data from Ames, Iowa.

    Enter values for various features and get an instant price prediction.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit.")

current_dir = os.getcwd()
model_path = os.path.join(current_dir, "outputs", "models", "best_model.pkl")
data_path = os.path.join(current_dir, "outputs", "datasets", "cleaned", "TrainSetCleaned.csv")


if not os.path.exists(model_path):
    st.error(f"Model not found at: {model_path}")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    import traceback
    st.error(f"Error loading model: {e}")
    traceback.print_exc()
    model = None
    st.stop()


if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.subheader("📊 Example of training data:")
    st.dataframe(df.head())
else:
    st.warning(f"Data file not found at: {data_path}")


st.subheader("🔍 Make Your Own Prediction:")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input(
        "Above ground living area (GrLivArea)",
        min_value=500, max_value=5000, value=1500
    )
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)

with col2:
    garage_area = st.slider("Garage Area (car capacity)", 0, 5, 2)
    total_bsmt_sf = st.number_input(
        "Basement square feet (TotalBsmtSF)",
        min_value=0, max_value=3000, value=800
    )


if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "OverallQual": overall_qual,
        "GarageArea": garage_area,
        "TotalBsmtSF": total_bsmt_sf
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted house price: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"Something went wrong with the prediction: {e}")
