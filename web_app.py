# -*- coding: utf-8 -*-
"""Untitled2.ipynb"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Page configuration
st.set_page_config(page_title="Web Interface Demo", layout="wide")

# Function to load data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar for file upload
st.sidebar.title("Upload a Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file).copy()
    st.title("Uploaded Dataset")
    st.write(data.head())

    # **Data Cleaning Options**

    # Replace 0 with NaN
    st.sidebar.subheader("Replace 0 with NaN (for numeric columns)")
    replace_zero = st.sidebar.checkbox("Replace 0 with NaN")
    if replace_zero:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = data[numeric_columns].replace(0, np.nan)
        st.info("Zeros in numeric columns have been treated as missing values.")

    # Fill missing values with mean
    st.sidebar.subheader("Handle Missing Values")
    fill_missing = st.sidebar.checkbox("Fill Missing Values with Mean")
    if fill_missing:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        st.success("Missing values in numeric columns filled with their mean.")

    # Remove duplicate rows
    st.sidebar.subheader("Removing Duplicates")
    remove_duplicates = st.sidebar.checkbox("Removing Duplicates")
    if remove_duplicates:
        data.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed")

    # **Outlier Detection**
    st.sidebar.subheader("Outlier Detection")
    detect_outliers = st.sidebar.checkbox("Detect and Cap Outliers")
    if detect_outliers:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers
            data[column] = np.where(data[column] > upper_bound, upper_bound,
                                    np.where(data[column] < lower_bound, lower_bound, data[column]))
        st.success("Outliers detected and capped.")

    # **Standardize Columns**
    st.sidebar.subheader("Standardize Numeric Columns")
    standardize_data = st.sidebar.checkbox("Standardize Numeric Columns")
    if standardize_data:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        st.success("Numeric columns standardized.")

    # **Display Cleaned Data**
    st.sidebar.subheader("Display Cleaned Data")
    show_cleaned_data = st.sidebar.checkbox("View Cleaned Data")
    if show_cleaned_data:
        st.write("### Cleaned Dataset")
        st.write(data.head(50))  # Display the first 50 rows
        st.write(f"Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")

    # **Download Cleaned Data**
    st.sidebar.subheader("Download Cleaned Dataset")
    download_button = st.sidebar.button("Download Cleaned Dataset as CSV")
    if download_button:
        st.download_button(
            label="Download CSV",
            data=data.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # EDA Options
    st.sidebar.subheader("Exploratory Data Analysis(EDA)")
    if st.sidebar.checkbox("Summary Statistics"):
        st.write("### Summary Statistics")
        st.write(data.describe())

    if st.sidebar.checkbox("Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Model Selection
    st.sidebar.subheader("Machine Learning Models")
    model_choice = st.sidebar.selectbox("Choose a model", 
                                        ["None", "Linear Regression", "K-Means Clustering", 
                                         "Decision Tree", "SVM Regression"])

    if model_choice == "Linear Regression":
        st.write("### Linear Regression")
        target = st.sidebar.selectbox("Target Variable", data.columns)
        features = st.sidebar.multiselect("Feature Variables", data.columns)

        if target and features:
            numeric_columns = data[features].select_dtypes(include=[np.number]).columns
            X = data[numeric_columns]
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            st.write("#### Model Performance")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R² Score: {reg.score(X_test, y_test):.2f}")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

    elif model_choice == "K-Means Clustering":
        st.write("### K-Means Clustering")
        features = st.sidebar.multiselect("Feature Variables", data.columns)

        if features:
            X = data[features]
            k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            data["Cluster"] = kmeans.fit_predict(X)

            st.write("#### Cluster Assignments")
            st.write(data[["Cluster"] + features].head())

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=data["Cluster"], palette="viridis", ax=ax)
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)

    elif model_choice == "Decision Tree Regression":
        st.write("### Decision Tree Regression")
        target = st.sidebar.selectbox("Target Variable", data.columns, key="dt_target")
        features = st.sidebar.multiselect("Feature Variables", data.columns, key="dt_features")

        if target and features:
            numeric_columns = data[features].select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                st.error("No numeric features selected. Please select valid numeric columns.")
            else:
                X = data[numeric_columns]
                y = data[target]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                dt_model = DecisionTreeRegressor(random_state=42)
                dt_model.fit(X_train, y_train)

                # Predictions
                y_pred = dt_model.predict(X_test)
    
                # Display performance metrics
                st.write("#### Model Performance")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

                # Visualization
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
        else:
            st.warning("Please select both target variable and feature variable.")

    # Similar validation for SVM Regression
    elif model_choice == "SVM Regression":
        st.write("### Support Vector Machine Regression")
        target = st.sidebar.selectbox("Target Variable", data.columns, key="svm_target")
        features = st.sidebar.multiselect("Feature Variables", data.columns, key="svm_features")

        if target and features:
            numeric_columns = data[features].select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                st.error("No numeric features selected. Please select valid numeric columns.")
            else:
                X = data[numeric_columns]
                y = data[target]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                svm_model = SVR(kernel="rbf")
                svm_model.fit(X_train, y_train)

                # Predictions
                y_pred = svm_model.predict(X_test)

                # Display performance metrics
                st.write("#### Model Performance")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

                # Visualization
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
        else:
            st.warning("Please select both target variable and feature variable.")








            # Visualization
