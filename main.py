import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Extract scaler and KMeans model
scaler = model["scaler"]
kmeans = model["model"]

# Load dataset
# data_path = "/mnt/data/rfm_df.csv"
df = pd.read_csv('rfm_df.csv')

# Ensure only relevant features are used
feature_columns = [col for col in df.columns if col not in ["Cluster", "customer_unique_id"]]
X = df[feature_columns]

# Get number of features
try:
    n_features = scaler.n_features_in_  # Coba ambil jumlah fitur dari scaler
except AttributeError:
    n_features = kmeans.cluster_centers_.shape[1]  # Ambil dari KMeans jika tidak ada di scaler

# Streamlit App Title
st.title("KMeans Clustering Dashboard")

# Create Tabs
tab1, tab2 = st.tabs(["Cluster Visualization", "Cluster Prediction"])

# Tab 1: Cluster Visualization
with tab1:
    st.header("Cluster Distribution")
    
    # Transform dataset using the scaler
    X_scaled = scaler.transform(X)
    cluster_labels = kmeans.predict(X_scaled)
    df["Cluster"] = cluster_labels

    # Show Cluster Distribution
    st.write("### Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Cluster"], palette="viridis", ax=ax)
    st.pyplot(fig)

    # Show Cluster Centers
    st.write("### Cluster Centers")
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert back to original scale
    st.dataframe(pd.DataFrame(cluster_centers, columns=feature_columns))

# Tab 2: Cluster Prediction
with tab2:
    st.header("Cluster Prediction UI")
    sub_tab1, sub_tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # Single Prediction
    with sub_tab1:
        st.subheader("Single Prediction")

        feature_input = {}
        for col in feature_columns:
            feature_input[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Predict Cluster"):
            input_df = pd.DataFrame([feature_input])
            scaled_input = scaler.transform(input_df)
            cluster_prediction = kmeans.predict(scaled_input)
            st.success(f"Predicted Cluster: {cluster_prediction[0]}")

    # Batch Prediction
    with sub_tab2:
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            batch_df = batch_df[feature_columns]  # Ensure only relevant features are used
            st.write("Uploaded Data:")
            st.dataframe(batch_df.head())

            scaled_batch = scaler.transform(batch_df)
            batch_clusters = kmeans.predict(scaled_batch)
            batch_df["Predicted Cluster"] = batch_clusters

            st.write("### Predictions")
            st.dataframe(batch_df)

            st.download_button("Download Cluster Predictions", batch_df.to_csv(index=False), "cluster_predictions.csv")
