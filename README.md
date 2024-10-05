# Skincare Product Review Rating Predictor

This project is a simple web app that predicts the rating (between 1 and 5) for a skincare product review using machine learning. The data was sourced from Kaggle and cleaned before building the model. The app is built using Streamlit and Python libraries like Scikit-learn.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [EDA and Preprocessing](#eda-and-preprocessing)
- [Model Training](#model-training)
- [Model Deployment with Streamlit](#model-deplyment-with-streamlit)
- [Running the App](#running-the-app)

## Overview
This app takes a skincare product review and predicts its rating. It uses a Logistic Regression model trained on a dataset of reviews. The app is designed to be lightweight and easy to use, allowing users to predict ratings by entering their review in a text box.

## Dataset
- **Source**: Kaggle
- **Product**: Lotus Balancing & Hydrating Natural Face Treatment Oil
- The dataset includes user reviews and their corresponding ratings. After removing rows with missing values, the focus was on two columns:
  - `review_text` (the actual review)
  - `rating` (the rating, between 1 and 5)

## EDA and Preprocessing
During Exploratory Data Analysis (EDA), the following steps were performed:
- **Remove Missing Data**: Any reviews with missing values were removed.
- **Text Preprocessing**: 
  - Convert to lowercase
  - Remove punctuation
  - Remove stopwords (like "the", "and"), but keep important words like "not"
  - Apply stemming and lemmatization to standardize words (e.g., "running" becomes "run")
  
 ## Model Training
A Logistic Regression model was trained on vectorized review data using the `CountVectorizer` from Scikit-learn. The training data was imbalanced (skewed), so the model was trained with `class_weight='balanced'` to account for the skewed distribution of ratings.

After training, the model achieved a validation accuracy of approximately 63.62% on the test set.

```python
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vectors, y_train)
```

## Model Deployment with Streamlit
The trained model was saved using pickle and deployed in a Streamlit app. Users can input a review into the app, which will preprocess the text, vectorize it, and predict a rating on a scale of 1 to 5.

How the App Works:
- User Input: The user enters a product review in a text box.
- Prediction: After clicking the "Predict" button, the model predicts the likely rating for the review.

## Running the App
You can run this app locally using Streamlit. Here’s how to set it up:

- Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```
- Install the dependencies:
```bash
pip install -r requirements.txt
```
- Run the Streamlit app:
```bash
streamlit run app.py
```

Once the app is running, you can enter a skincare review into the text box, click "Predict," and the app will return a predicted rating.

