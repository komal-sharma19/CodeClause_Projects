import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Customer Segmentation Tool", layout="wide")

# Title
st.title("Customer Segmentation Tool")
st.write("Upload E-Commerce customer data, perform K-Means clustering, and visualize customer segments.")

# File uploader
uploaded_file = st.file_uploader("Upload E-Commerce Data CSV", type=["csv"])

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Data Cleaning
    st.subheader("Step 1: Data Cleaning")
    st.write("Removing null CustomerIDs, canceled orders, and negative/zero quantities and prices.")

    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Feature Engineering
    st.subheader("Step 2: Feature Engineering")
    customer_df = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceNo': 'NumTransactions',
        'Quantity': 'TotalQuantity',
        'TotalPrice': 'TotalSpend'
    }).reset_index()

    st.dataframe(customer_df.head())

    # Normalization
    st.subheader("Step 3: Feature Normalization")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_df[['NumTransactions', 'TotalQuantity', 'TotalSpend']])
    st.write("Features scaled using StandardScaler.")

    # Cluster selection
    st.subheader("Step 4: K-Means Clustering")
    k = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=4)

    if st.button("Run Clustering"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.success(f"Clustering completed with {k} clusters.")
        st.dataframe(customer_df.head())

        # Download clustered data
        csv = customer_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data", csv, "clustered_customers.csv", "text/csv")

        # Visualization
        st.subheader("Step 5: Cluster Visualization")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(
            x=customer_df['TotalSpend'],
            y=customer_df['NumTransactions'],
            hue=customer_df['Cluster'],
            palette='tab10',
            ax=ax1
        )
        ax1.set_title("Total Spend vs. Number of Transactions")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            x=customer_df['TotalSpend'],
            y=customer_df['TotalQuantity'],
            hue=customer_df['Cluster'],
            palette='tab10',
            ax=ax2
        )
        ax2.set_title("Total Spend vs. Total Quantity")
        st.pyplot(fig2)

else:
    st.info("Please upload your Kaggle E-Commerce dataset CSV to proceed with customer segmentation.")
