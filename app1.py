import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def app():
        
    # App Title with improved styling
    st.title("🎓 Student Performance Prediction")
    st.write("This app predicts students' performance using various regression models. You can select different models and visualize how well they predict student scores based on their features.")

    # Theoretical Details about the app's functionality
    st.write("""
    ### App Overview:
    This app uses various regression models to predict the academic performance of students, specifically their percentage. 
    We provide three machine learning models for prediction:
    - **Linear Regression**: A basic algorithm for predicting continuous variables based on linear relationships.
    - **Random Forest**: A robust ensemble learning method that works by combining several decision trees to improve predictions.
    - **Support Vector Regressor (SVR)**: A method that works by finding the best hyperplane to predict continuous values in high-dimensional spaces.

    The model will be trained on student data and its performance will be evaluated using metrics like **Mean Squared Error** and **R-squared**.

    ### Steps:
    1. **Dataset Loading**: The student data is loaded from a CSV file.
    2. **Data Preprocessing**: The data is standardized for better performance in some models.
    3. **Model Training**: You can select any model from the options provided.
    4. **Model Evaluation**: The app will display metrics to evaluate the model's performance.
    """)

    # Load existing CSV file
    data_file = 'student_data.csv'
    data = pd.read_csv(data_file)

    # Display dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head(), width=700)

    # Split data into features and target
    X = data.drop('Percentage', axis=1)
    y = data['Percentage']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection with more styling
    model_choice = st.selectbox(
        "Select a model for prediction:", 
        ("Linear Regression", "Random Forest", "Support Vector Regressor"),
        index=0,
        help="Choose a regression model to predict student performance."
    )

    # Model choice and training
    if model_choice == "Linear Regression":
        model = LinearRegression()
        st.write("""
        **Linear Regression**:
        Linear regression attempts to model the relationship between two variables by fitting a linear equation to the observed data.
        It assumes a linear relationship between the dependent variable and the independent variables.
        """)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        st.write("""
        **Random Forest**:
        Random Forest is an ensemble learning method that constructs multiple decision trees and merges their results to improve prediction accuracy.
        It is robust against overfitting and works well for complex datasets.
        """)
    else:
        model = SVR()
        st.write("""
        **Support Vector Regressor (SVR)**:
        SVR tries to fit the best possible hyperplane that can predict the target variable while keeping the margin as large as possible.
        It is particularly effective for high-dimensional spaces and when there is non-linear behavior in the data.
        """)

    # Train the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.write(f"### Model: {model_choice}")
    st.write(f"- **Mean Squared Error**: {mse:.2f} (Lower values are better)")
    st.write(f"- **R-squared**: {r2:.2f} (Higher values indicate better model performance)")

    # Aesthetic scatter plot with additional information
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.6, edgecolors='w', label='Predictions', color='cornflowerblue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual Percentage", fontsize=12)
    ax.set_ylabel("Predicted Percentage", fontsize=12)
    ax.set_title(f"{model_choice} Predictions vs Actual", fontsize=14)
    ax.legend()

    # Show the plot in Streamlit app
    st.pyplot(fig)

    # Add footer with more information
    st.write("""
    ---
    Developed by: Jagrut Thakare | Student Performance Prediction App | Powered by Streamlit
    """)
