# House Price Prediction in Ames, Iowa

## Project Overview

This project aims to predict the sale prices of houses in Ames, Iowa, based on their attributes using machine learning and predictive analytics. The dataset contains house attributes such as floor area, garage size, and condition, and their corresponding sale prices. The goal is to create a model that can predict sale prices with a minimum R² score of 0.75 and develop a dashboard that visualizes the results for the client’s inherited properties as well as other properties in Ames.

## How to Use This Repo

1. Clone the repository to your local machine or use GitHub Codespaces to open it in an IDE.
2. Install the necessary dependencies by running the following command:
pip3 install -r requirements.txt

1.Open the Jupyter notebook directory and start analyzing the dataset or training the model.
2 Run the scripts for data cleaning, feature engineering, model training, and validation.

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). It includes housing records from Ames, Iowa, representing house profiles (floor area, basement, garage, kitchen, lot size, year built, etc.) and their sale prices. The dataset includes the following columns:

| Variable        | Meaning                                          | Units               |
|-----------------|--------------------------------------------------|---------------------|
| 1stFlrSF        | First floor square feet                          | 334 - 4692 sq ft    |
| 2ndFlrSF        | Second-floor square feet                         | 0 - 2065 sq ft      |
| BedroomAbvGr    | Bedrooms above grade                             | 0 - 8               |
| BsmtExposure    | Exposure of basement to the outside              | Gd, Av, Mn, No      |
| ...             | ...                                              | ...                 |
| SalePrice       | Sale Price                                       | 34,900 - 755,000 USD|

## Business Requirements

Your friend, who inherited properties in Ames, Iowa, needs help predicting the sale prices for her four inherited houses. Although she has local knowledge, she fears her appraisals may be inaccurate due to differing market conditions in Ames. The following requirements are provided:

1. **Discover house attributes' correlations with sale prices**: You are expected to visualize how various attributes correlate with the sale price.
2. **Predict sale price for inherited houses and other properties**: The model should predict the sale prices for your friend's inherited properties, as well as any other properties in Ames.

## Hypothesis and Validation

- **Hypothesis**: House features such as square footage, number of bedrooms, garage area, and overall condition influence the sale price.
- **Validation**: Correlate these features with the sale price using data visualizations. Train the model using these features and evaluate it using metrics like R².

## Rationale for Data Visualizations and ML Tasks

To answer the business requirements, we will perform the following tasks:

1. **Data Visualization**: Use scatter plots, heatmaps, and bar charts to visualize correlations between features and sale prices.
2. **Machine Learning**: Use regression models (e.g., Linear Regression, Random Forest) to predict sale prices based on house features.

## ML Business Case

The machine learning model will help predict the sale price of the properties, ensuring accurate appraisals for the inherited houses. It will also help provide a market overview of property prices in Ames, Iowa, assisting in the decision-making process regarding property sales.

## Dashboard Design

The dashboard will consist of the following pages and content:

1. **Project Summary Page**: Overview of the project, objectives, and methodology.
2. **Feature Correlation Page**: Visualize the relationship between different house attributes and sale price.
3. **Predicted Prices Page**: Display the predicted sale prices for the inherited properties and other properties.
4. **Hypothesis Validation Page**: Provide insights into the validation of the project hypotheses.
5. **Model Performance Page**: Display model metrics, including the R² score, and visualize predictions versus actual prices.

## Unfixed Bugs

- **Bug 1**: Some visualizations may not display correctly on the dashboard if the data is not preprocessed correctly.
- **Bug 2**: Model accuracy may vary depending on feature selection, and further tuning of hyperparameters may be required.

## Deployment

### Heroku

- The live app link is: [https://YOUR_APP_NAME.herokuapp.com/](https://YOUR_APP_NAME.herokuapp.com/)
- Set the .python-version` Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack supported version.
- To deploy the app to Heroku, follow these steps:
  
1. Log in to Heroku and create an app.
2. Under the Deploy tab, select GitHub as the deployment method.
3. Search for the repository and click Connect.
4. Choose the branch to deploy and click Deploy Branch.
5. Once the deployment process completes, click Open App to view your app.

## Main Data Analysis and Machine Learning Libraries

- **pandas**: Used for data manipulation and cleaning.
- **matplotlib**: Used for data visualization (scatter plots, heatmaps, etc.).
- **seaborn**: For enhanced visualizations like heatmaps for correlation analysis.
- **scikit-learn**: For machine learning algorithms (Linear Regression, Random Forest, etc.) and evaluation metrics.
- **Jupyter Notebook**: Used for running and testing code interactively.

## Credits

### Content

- The dataset used in this project was taken from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).
- Instructions for deploying the app to Heroku were taken from [Heroku Documentation](https://devcenter.heroku.com/articles/git).
- Dashboard and visualizations were inspired by [Data Science Tutorials](https://www.datacamp.com/community/tutorials).

## Acknowledgements
