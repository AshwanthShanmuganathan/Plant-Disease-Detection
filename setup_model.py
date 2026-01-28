import os
import tensorflow as tf
from model import PlantDiseaseModel
import gdown

def download_model():
    """Download the pre-trained model from Google Drive"""
    model_path = os.path.join('models', 'plant_disease_model.h5')
    
    if not os.path.exists(model_path):
        print("Downloading pre-trained model...")
        # You would normally put your own pre-trained model URL here
        # This is a placeholder URL
        url = 'YOUR_GDRIVE_MODEL_URL'
        gdown.download(url, model_path, quiet=False)
    else:
        print("Model already exists")

def main():
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download the pre-trained model
    download_model()
    
    # Test model loading
    try:
        model = PlantDiseaseModel()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
