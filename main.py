import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('model3.pkl')

pipeline = load_model()
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']

# Get number of features from the scaler
n_features = scaler.n_features_in_

# Create tabs
tab1, tab2 = st.tabs(["Model Performance", "Model Prediction"])

# Tab 1 - Model Performance Visualization
with tab1:
    st.header("Model Performance Metrics")
    
    # Display basic model information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Clusters", model.n_clusters)
    with col2:
        st.metric("Inertia", f"{model.inertia_:.2f}")
    
    # Visualization of cluster centers
    st.subheader("Cluster Centers Visualization")
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    cluster_centers_2d = pca.fit_transform(model.cluster_centers_)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], 
                        c=range(model.n_clusters), cmap='viridis', s=200)
    ax.set_title("Cluster Centers (PCA Reduced)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(fig)

# Tab 2 - Model Prediction UI
with tab2:
    st.header("Prediction Interface")
    
    prediction_type = st.radio("Select Prediction Type:", 
                             ("Single Prediction", "Batch Prediction"))
    
    if prediction_type == "Single Prediction":
        st.subheader("Single Data Point Prediction")
        inputs = []
        
        # Create input fields dynamically based on number of features
        cols = st.columns(3)
        for i in range(n_features):
            with cols[i % 3]:
                inputs.append(st.number_input(f"Feature {i+1}", value=0.0))
        
        if st.button("Predict Cluster"):
            # Prepare input data
            input_data = np.array(inputs).reshape(1, -1)
            # Transform using pipeline
            prediction = pipeline.predict(input_data)
            st.success(f"Predicted Cluster: {prediction[0]}")
            
    else:
        st.subheader("Batch Prediction from CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if len(df.columns) != n_features:
                    st.error(f"Invalid number of features. Expected {n_features} columns.")
                else:
                    # Make predictions
                    predictions = pipeline.predict(df)
                    df['Cluster'] = predictions
                    
                    st.subheader("Prediction Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        data=csv,
                        file_name="cluster_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")