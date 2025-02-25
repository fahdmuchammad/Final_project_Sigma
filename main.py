import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA

@st.cache_resource
def load_model():
    return joblib.load('model1.pkl')

pipeline = load_model()

def determine_feature_count(pipeline):
    # Try common attributes first
    try:
        if hasattr(pipeline, 'n_features_in_'):
            return pipeline.n_features_in_
        if 'scaler' in pipeline.named_steps:
            return pipeline.named_steps['scaler'].scale_.shape[0]
        return pipeline.named_steps['model'].cluster_centers_.shape[1]
    except AttributeError:
        pass

    # Brute-force dimension detection
    for n in range(1, 100):
        try:
            dummy_data = np.zeros((1, n))
            pipeline.predict(dummy_data)
            return n
        except (ValueError, TypeError) as e:
            error_msg = str(e)
            if "feature" in error_msg and "expected" in error_msg:
                match = re.search(r'expected (\d+) features', error_msg)
                if match:
                    return int(match.group(1))
    
    # Final fallback
    st.error("""Could not automatically determine feature count. 
             Please enter manually:""")
    return st.number_input("Number of features in model", 
                          min_value=1, value=4, step=1)

n_features = determine_feature_count(pipeline)

# Rest of your original tab code remains the same...
# Create tabs
tab1, tab2 = st.tabs(["Model Performance", "Model Prediction"])

# Tab 1 - Model Performance Visualization
with tab1:
    st.header("Model Performance Metrics")
    
    # Display basic model information
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.metric("Number of Clusters", model.n_clusters)
        except:
            st.warning("Could not retrieve cluster count")
    
    # Visualization section remains similar but with error handling...

# Tab 2 - Model Prediction UI
with tab2:
    st.header("Prediction Interface")
    
    prediction_type = st.radio("Select Prediction Type:", 
                             ("Single Prediction", "Batch Prediction"))
    
    if prediction_type == "Single Prediction":
        st.subheader("Single Data Point Prediction")
        inputs = []
        
        # Create input fields based on determined features
        cols = st.columns(3)
        for i in range(n_features):
            with cols[i % 3]:
                inputs.append(st.number_input(f"Feature {i+1}", value=0.0))
        
        if st.button("Predict Cluster"):
            try:
                input_data = np.array(inputs).reshape(1, -1)
                prediction = pipeline.predict(input_data)
                st.success(f"Predicted Cluster: {prediction[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                
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