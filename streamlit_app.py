import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

def main():
    st.title("Student Performance Prediction")

    # Allow the user to upload the CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)

        # Clean and preprocess the data
        # Handle missing values
        print(df.isnull().sum())
        df = df.fillna(df.mean())

        # Handle outliers (optional)
        z = np.abs(stats.zscore(df))
        threshold = 3
        df = df[(z < threshold).all(axis=1)]

        # Encode categorical variables (if any)
        # Example: Encode a categorical feature 'gender'
        # from sklearn.preprocessing import LabelEncoder
        # le = LabelEncoder()
        # df['gender'] = le.fit_transform(df['gender'])

        # Split the data into features and target
        X = df[['feature1', 'feature2', 'feature3']]  # Replace with your feature names
        y = df['math_score']

        # Split the data into training and testing sets (optional)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform multilinear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        r_squared = model.score(X_test, y_test)
        print(f"R-squared: {r_squared:.2f}")

        # Create the Streamlit application
        st.subheader("Data Cleaning")
        st.write("Summary of missing values:")
        st.write(df.isnull().sum())

        st.subheader("Data Visualization")
        fig, ax = plt.subplots()
        ax.hist(df['math_score'], bins=20)
        ax.set_xlabel('Math Score')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        st.subheader("Make a Prediction")
        feature1 = st.number_input("Feature 1", value=0.0)
        feature2 = st.number_input("Feature 2", value=0.0)
        feature3 = st.number_input("Feature 3", value=0.0)

        if st.button("Predict"):
            prediction = model.predict([[feature1, feature2, feature3]])
            st.write(f"Predicted Math Score: {prediction[0]:.2f}")

        st.subheader("Scatter Plot")
        fig, ax = plt.subplots()
        ax.scatter(X_train['feature1'], y_train, label='Training Data')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Math Score')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
