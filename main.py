import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
import seaborn as sns
import squarify
import plotly.express as px

# Function for load model
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

pipeline_main = load_model('model3.pkl')
pipeline_fallback = load_model('model2_cluster1.pkl')

@st.cache_data
def load_dataset():
    df = pd.read_csv("rfm_df1.csv")
    return df

df = load_dataset()

# Pmake sure feature selection
expected_features = pipeline_main.named_steps['scaler'].feature_names_in_
df = df[expected_features]

# Ttransform dataset with scaler from model
X_scaled = pipeline_main.named_steps['scaler'].transform(df)

# Compute clustering metrics (cache results to avoid recalculating them over and over)
@st.cache_data
def compute_clustering_metrics(X_scaled, _model_labels):
    silhouette_avg = silhouette_score(X_scaled, _model_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, _model_labels)
    return silhouette_avg, davies_bouldin

silhouette_avg, davies_bouldin = compute_clustering_metrics(
    X_scaled, pipeline_main.named_steps['model'].labels_
)


# silhouette_avg, davies_bouldin = compute_clustering_metrics(X_scaled, pipeline_main.named_steps['model'])

# Visualization of cluster centers with PCA (cache to avoid repeated calculations)
@st.cache_data
def compute_pca_clusters(cluster_centers):
    pca = PCA(n_components=2)
    return pca.fit_transform(cluster_centers)

cluster_centers = pipeline_main.named_steps['model'].cluster_centers_
cluster_centers_2d = compute_pca_clusters(cluster_centers)

# Create tabs
tab1, tab2 = st.tabs(["Model Performance", "Model Prediction"])

# Tab 1 - Model Performance Visualization
with tab1:
    st.header("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
    with col2:
        st.metric("Davies-Bouldin Score", f"{davies_bouldin:.3f}")

    # Plot Cluster Centers
    st.title("Customer Segmentation Grid")

    # Upload file CSV
    uploaded_file = st.file_uploader("Upload your RFM dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check that all required columns are present
        required_columns = {"customer_unique_id", "frequency", "monetary", "recency", "Cluster"}
        if not required_columns.issubset(df.columns):
            st.error(f"Dataset harus memiliki kolom: {required_columns}")
        else:
            # Calculate the total monetary and average frequency per cluster.
            cluster_summary = df.groupby("Cluster").agg({
                "monetary": "sum",
                "frequency": "mean"
            }).reset_index()

            # The order so that Best Customers appear in the top right corner
            sort_order = ["Lost Customer", "Potential Customer", "New Customers", "At Risk Customers", "Best Customers"]
            cluster_summary["SortOrder"] = cluster_summary["Cluster"].apply(lambda x: sort_order.index(x))
            cluster_summary = cluster_summary.sort_values("SortOrder")

            # Mapping color based on cluster
            color_map = {
                "Lost Customer": "red",
                "Potential Customer": "orange",
                "New Customers": "blue",
                "At Risk Customers": "yellow",
                "Best Customers": "fuchsia"
            }

            # Create colors based on cluster order
            colors = [color_map[c] for c in cluster_summary["Cluster"]]

            # Treemap Plot using Squarify
            fig, ax = plt.subplots(figsize=(12, 7))
            squarify.plot(
                sizes=cluster_summary["monetary"], 
                label=cluster_summary["Cluster"], 
                color=colors, 
                alpha=0.7,
                text_kwargs={'fontsize': 14}
            )

            plt.title("Customer Segmentation Grid", fontsize=20)
            plt.axis('off')  # Remove the wick for a cleaner look

            # print plot on Streamlit
            st.pyplot(fig)

    

# Tab 2 - Model Prediction UI
with tab2:
    st.header("Prediction Interface")
    
    prediction_type = st.radio("Select Prediction Type:", 
                             ("Single Prediction", "Batch Prediction"))
    
    if prediction_type == "Single Prediction":
        st.subheader("Single Data Point Prediction")
        inputs = []
        # Input fields for the main model
        cols = st.columns(3)
        for i, feature in enumerate(expected_features):
            with cols[i % 3]:
                inputs.append(st.number_input(feature, value=0.0, key=f"main_{i}"))
        
        if st.button("Predict Cluster"):
            input_data = np.array(inputs).reshape(1, -1)
            # Take the index of the columns "recency", "monetary", and "frequency"
    
            if np.all(input_data == 0):
                st.error("⚠️ please dont fill with 0 value. input valid number.")
            
            # Prediction with the main model
            else:
                main_prediction = pipeline_main.predict(input_data)[0]
                
                if main_prediction == 1:  # If the prediction cluster 1
                    st.warning("Cluster 1 detected, run to second model")
                    
                    # Prediction with secondary models
                    fallback_prediction = pipeline_fallback.predict(input_data[:, :len(expected_features)])[0]
                    st.success(f"Final Predict Result: Cluster 1 - Subcluster {fallback_prediction}")
                    if fallback_prediction == 0:
                        st.title("Lost Customer")
                        st.header('Recommendation')
                        st.success('Focus on re-engagement with special discounts, promotions, and personalized follow-ups.')
                        st.success('Use targeted email campaigns to reconnect with these customers and remind them of your products/services.')
                    else:
                        st.title("Potential Customer")
                        st.header('Recommendation')
                        st.success('Increase engagement through personalized campaigns, exclusive offers, and time-limited promotions to boost repeat purchases.')
                        st.success('Create urgency with limited-time deals to encourage immediate action.')
                else:
                    st.success(f"Predict Result: Cluster {main_prediction}")
                    if main_prediction == 0:
                        st.title("At Risk Customer")
                        st.header('Recommendation')
                        st.success('Launch targeted campaigns with discounts, loyalty rewards, and personalized emails to re-engage them.')
                        st.success('Tailor campaigns based on product categories they frequently buy (e.g., bed_bath_table, furniture_decor, computers_accessories).')
                    elif main_prediction == 2:
                        st.title("New Customer")
                        st.header('Recommendation')
                        st.success('Nurture them with personalized post-purchase follow-ups, targeted offers, and loyalty incentives to encourage repeat purchases.')
                        st.success('Set up a referral program to incentivize new customers to refer others, increasing their lifetime value.')
                    else:
                        st.title("Best Customer")
                        st.header('Recommendation')
                        st.success('Retain these high-value customers with exclusive offers, VIP rewards, and early access to new products.')
                        st.success('Show appreciation with personalized experiences to deepen their loyalty.')
            
    else:
        st.subheader("Batch Prediction from CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)

            # Make sure the number of features is appropriate
            if set(expected_features) != set(df_uploaded.columns):
                st.error("Kolom dalam file tidak sesuai dengan fitur yang digunakan saat training.")
            else:
                # Prediction with the main model
                X_scaled_uploaded = pipeline_main.named_steps['scaler'].transform(df_uploaded)
                main_predictions = pipeline_main.predict(X_scaled_uploaded)
                df_uploaded['Main Cluster'] = main_predictions
                
                # Prediction for cluster 1 with secondary model
                cluster1_mask = df_uploaded['Main Cluster'] == 1
                if cluster1_mask.any():
                    cluster1_data = df_uploaded.loc[cluster1_mask, expected_features[:len(expected_features)]]
                    fallback_predictions = pipeline_fallback.predict(cluster1_data)
                    df_uploaded.loc[cluster1_mask, 'Subcluster'] = fallback_predictions
                
                st.subheader("Hasil Prediksi")
                st.dataframe(df_uploaded)
                
                csv = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Hasil",
                    data=csv,
                    file_name="cluster_predictions.csv",
                    mime="text/csv"
                )
