import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import chardet

# Page Configuration
st.set_page_config(page_title="Data Visualization and Cleaning", layout="wide")

# Function to detect file encoding
@st.cache
def detect_encoding(file):
    # Read the first 10,000 bytes of the file to detect encoding
    raw_data = file.read(10000)
    file.seek(0)  # Reset file pointer to the start
    result = chardet.detect(raw_data)
    return result['encoding']

# Function to load data
@st.cache
def load_data(file, encoding):
    try:
        data = pd.read_csv(file, encoding=encoding)
        return data
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

# Sidebar for file upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Detect encoding
    detected_encoding = detect_encoding(uploaded_file)
    st.sidebar.write(f"Detected Encoding: {detected_encoding}")
    encoding_option = st.sidebar.selectbox(
        "Select File Encoding", [detected_encoding, "utf-8", "ISO-8859-1", "cp1252"]
    )

    # Load data
    data = load_data(uploaded_file, encoding_option)

    if data is not None:
        st.title("Uploaded Dataset")
        st.write(data.head())

        # Data Cleaning
        st.subheader("Data Cleaning")
        missing_values = data.isnull().sum()
        st.write("Missing Values:")
        st.write(missing_values[missing_values > 0])

        # Dropping unnecessary columns (user-defined)
        drop_columns = st.sidebar.multiselect("Select columns to drop", data.columns)
        if drop_columns:
            data.drop(columns=drop_columns, inplace=True)
            st.write(f"Columns dropped: {drop_columns}")

        # Handle duplicates
        if st.sidebar.checkbox("Remove Duplicates"):
            data.drop_duplicates(inplace=True)
            st.write("Duplicates removed")

        # Convert date columns
        date_columns = st.sidebar.multiselect("Select date columns to convert", data.columns)
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            st.write(f"Converted {col} to datetime")

        # Normalizing text columns
        text_columns = st.sidebar.multiselect("Select text columns to normalize", data.columns)
        for col in text_columns:
            data[col] = data[col].str.lower().str.strip()
            st.write(f"Normalized text in {col}")

        # Data Overview Post Cleaning
        st.subheader("Data Overview Post Cleaning")
        st.write(data.describe())

        # Visualizations
        st.header("Visualizations")

        # Heatmap
        if st.sidebar.checkbox("Show Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                corr = numeric_data.corr()
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.write("No numeric columns available for heatmap.")

        # Histograms
        if st.sidebar.checkbox("Show Histograms"):
            st.subheader("Histograms")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                plt.figure()
                sns.histplot(data[col], kde=True)
                plt.title(f"Distribution of {col}")
                st.pyplot(plt)

        # Modeling (Linear Regression Example)
        st.header("Modeling")
        target = st.sidebar.selectbox("Select target variable", data.columns)
        features = st.sidebar.multiselect("Select feature variables", data.columns)

        if target and features:
            st.subheader("Linear Regression Model")
            X = data[features]
            y = data[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            st.write("Model Coefficients:")
            st.write(model.coef_)
            st.write("R² Score:")
            st.write(model.score(X_test, y_test))

            # Visualization
            plt.figure()
            plt.scatter(y_test, y_pred)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            st.pyplot(plt)
    else:
        st.error("Failed to load the dataset. Please check the file encoding or format.")
else:
    st.write("Please upload a CSV file to begin.")
