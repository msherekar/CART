import streamlit as st
import requests
import os

# Title
st.title("CART Cytotoxicity Predictor")
st.write("Upload your CAR protein sequence in FASTA format to predict cytotoxicity scores!")

# File uploader
uploaded_file = st.file_uploader("Choose a FASTA file", type=["fasta", "fa", "txt"])

# Prediction button
if uploaded_file is not None:
    sequence = uploaded_file.read().decode('utf-8')
    
    # Description field
    description = st.text_input("Enter a description (optional):")
    
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            # Get API URL from environment variable or use default
            api_url = os.getenv('API_URL', 'http://localhost:8000/predict')
            payload = {"sequence": sequence, "description": description}
            
            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Show results
                st.success("Prediction completed!")
                st.write(f"**High Model Score:** {result['high_model_score']:.4f}")
                st.write(f"**Low Model Score:** {result['low_model_score']:.4f}")
                st.write(f"**Average Score:** {result['average_score']:.4f}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Prediction failed: {e}")
