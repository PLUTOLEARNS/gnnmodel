import streamlit as st
from leukemia_gnn import LeukemiaGNN, predict_leukemia
import torch
import os

def load_trained_model(model_path):
    model = LeukemiaGNN(num_features=1)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['val_acc']
    return None, None

def main():
    st.title("Leukemia Type Prediction")
    
    # Model loading
    model_path = "best_model.pth"
    model, val_acc = load_trained_model(model_path)
    
    if model is None:
        st.error("No trained model found. Please train the model first.")
        return
    
    st.info(f"Model loaded successfully! Validation accuracy: {val_acc:.2%}")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Blood Smear Image", use_column_width=True)
        
        # Add button to make prediction
        if st.button("Predict"):
            try:
                prediction, probability = predict_leukemia(model, uploaded_file)
                st.success(f"Prediction: {prediction}")
                st.info(f"Confidence: {probability:.2%}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()