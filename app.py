import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


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
    st.stop()


if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None
    st.warning(f"Data file not found at: {data_path}")

menu = st.sidebar.radio(
    "Go to:",
    ["Project Summary", "Feature Correlation", "Predicted Prices", "Hypothesis Validation", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit.")


if menu == "Project Summary":
    st.title("📑 Project Summary")
    st.write("""
    This app uses a regression model to predict house prices in Ames, Iowa.

    You can explore the training data, see correlations between variables, test hypotheses, and make your own predictions.
    """)
    if df is not None:
        st.subheader("📊 Example of training data:")
        st.dataframe(df.head())


elif menu == "Feature Correlation":
    st.title("📊 Feature Correlation")
    if df is not None:
        st.write("Correlation matrix between selected features and SalePrice:")
        corr = df[["SalePrice", "GrLivArea", "OverallQual", "GarageArea", "TotalBsmtSF"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data file found to display correlation.")


elif menu == "Predicted Prices":
    st.title("🔍 Make Your Own Prediction")

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
            st.error(f"Prediction failed: {e}")


elif menu == "Hypothesis Validation":
    st.title("📈 Hypothesis Validation")
    if df is not None:
        st.write("Hypothesis: The larger the living area, the higher the price.")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", ax=ax)
        ax.set_title("GrLivArea vs SalePrice")
        st.pyplot(fig)
    else:
        st.warning("No data file found for hypothesis validation.")


elif menu == "Model Performance":
    st.title("📊 Model Performance")
    if df is not None:
        X = df[["GrLivArea", "OverallQual", "GarageArea", "TotalBsmtSF"]]
        y = df["SalePrice"]
        y_pred = model.predict(X)
        score = r2_score(y, y_pred)
        st.write(f"R²-score on training data: **{score:.2f}**")
    else:
        st.warning("No data file found to calculate performance.")
