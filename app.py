import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor


notebook_path = "/workspaces/housingprices/jupyter_notebooks/modeling_evaluation_predict_price.ipynb"
data_path = "/workspaces/housingprices/outputs/datasets/cleaned/TrainSetCleaned.csv"
model_path = "/workspaces/housingprices/outputs/models/best_model.pkl"


if os.path.exists(notebook_path):
    with open(notebook_path, "r") as f:
        nb_content = nbformat.read(f, as_version=4)


    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb_content, {'metadata': {'path': '/workspaces/housingprices/jupyter_notebooks/'}})
        st.success("Notebook executed successfully!")
    except Exception as e:
        st.error(f"Error executing notebook: {e}")
        st.stop()
else:
    st.error(f"The notebook {notebook_path} doesn't exist. Please check the file path.")
    st.stop()


if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"The model file {model_path} doesn't exist. Please check the file path.")
    st.stop()

if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    st.error(f"The data file {data_path} doesn't exist. Please ensure the data is saved correctly.")
    st.stop()

st.title("Heritage Housing Price Predictor")
st.write("""
    This app allows you to input various house attributes, such as the overall quality, 
    living area, garage size, basement area, and year built, and predict the sale price 
    of a house in Ames, Iowa. The model was trained using a dataset of real estate data.
""")

st.sidebar.header("Enter house features")

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
    try:
        prediction = model.predict(input_df)  
        st.success(f"Predicted Sale Price: ${int(prediction[0]):,}")  #
    except Exception as e:
        st.error(f"Error making prediction: {e}")


if 'SalePrice' in data.columns:
    st.header("Correlation Heatmap")
    feature = st.selectbox("Choose feature to compare with SalePrice", data.columns)

    corr = data.corr(numeric_only=True)
    if feature in corr.columns:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr[[feature]].sort_values(by=feature, ascending=False), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting correlation heatmap: {e}")
    else:
        st.error(f"Feature {feature} not found in the data.")
else:
    st.error("The 'SalePrice' column is not found in the data.")
