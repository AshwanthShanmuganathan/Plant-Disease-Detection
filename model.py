import tensorflow as tf
import numpy as np
import os
from keras.applications import EfficientNetB3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model, load_model
from PIL import Image
from dataset_utils import PlantDiseaseDataset

class PlantDiseaseModel:
    def __init__(self, num_classes=None):
        self.img_size = (512, 512)
        
        # Define classes based on your trained model
        self.upload_classes = [
            'Tomato___healthy',
            'Tomato___Late_blight',
            'Tomato___Early_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Apple___healthy',
            'Apple___Apple_scab',
            'Apple___Black_rot'
        ]

        # Define potato-specific classes for camera capture
        self.capture_classes = [
            'Potato___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___Black_scurf',
            'Potato___Common_scab',
            'Potato___Ring_rot',
            'Potato___Soft_rot',
            'Potato___Leaf_roll_virus',
            'Potato___Mosaic_virus',
            'Potato___Root_knot',
            'Potato___Blackleg'
        ]
        
        # Use appropriate class list based on source type
        self.classes = self.upload_classes
        self.num_classes = len(self.classes)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load model directly from file
        model_path = os.path.join('models', 'best_model.h5')
        try:
            # Custom load function to handle incompatible layer names
            tf.keras.backend.clear_session()
            with tf.keras.utils.custom_object_scope({'CustomModel': Model}):
                self.model = load_model(model_path, compile=False)
            
            # Recompile the model with appropriate settings
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"Successfully loaded model from {model_path}")
            
            # Verify model architecture
            print("\nModel Architecture:")
            self.model.summary()
            
            # Verify output shape matches number of classes
            output_shape = self.model.output_shape
            if output_shape[-1] != self.num_classes:
                print(f"Warning: Model output shape {output_shape[-1]} doesn't match number of classes {self.num_classes}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting to build new model...")
            self.model = self.build_model()
            
    def build_model(self):
        """Build model with reduced complexity for faster training"""
        # Initialize the EfficientNetB3 model with imagenet weights
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(512, 512, 3)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        # Simplified dense layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Use legacy optimizer to avoid warnings
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def predict(self, image_path, source_type='uploaded'):
        """Make prediction with detailed plant information"""
        try:
            # Validate input
            if not os.path.exists(image_path):
                return {
                    'error': f'Image file not found: {image_path}',
                    'confidence': 0.0
                }
            
            # Validate image file type
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
                return {
                    'error': f'Invalid image file type. Supported types: {", ".join(valid_extensions)}',
                    'confidence': 0.0
                }

            # For captured images, always use potato classes
            if source_type == 'captured':
                self.classes = self.capture_classes
            else:
                self.classes = self.upload_classes
                
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            
            # Check image dimensions and resize if too large
            max_dimension = 4096  # Maximum allowed dimension
            if img.size[0] > max_dimension or img.size[1] > max_dimension:
                aspect_ratio = img.size[0] / img.size[1]
                if img.size[0] > img.size[1]:
                    new_size = (max_dimension, int(max_dimension / aspect_ratio))
                else:
                    new_size = (int(max_dimension * aspect_ratio), max_dimension)
                img = img.resize(new_size, Image.LANCZOS)
            
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get all prediction indices sorted by confidence
            sorted_indices = np.argsort(predictions[0])[::-1]
            
            if source_type == 'captured':
                # For captured images, always use potato predictions
                predicted_class = sorted_indices[0]
                # Force result to be a potato disease
                result = {
                    'plant_name': 'Potato',
                    'scientific_name': 'Solanum tuberosum',
                    'disease': self.capture_classes[predicted_class].split('___')[1].replace('_', ' '),
                    'confidence': float(predictions[0][predicted_class]),
                    'image_path': image_path,
                    'source_type': 'captured'
                }
                
                # Get treatment recommendations
                dataset = PlantDiseaseDataset()
                disease_info = dataset.get_disease_info(predicted_class)
                result.update({
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention']
                })
                
                # Add potato-specific information
                result.update({
                    'variety': 'Unknown - Please consult with local agricultural expert',
                    'growth_stage': 'Analysis based on leaf symptoms',
                    'additional_info': 'Potato plants require consistent moisture and well-drained soil'
                })
                
                # Print debug information for potato diseases only
                print(f"\nPotato Disease Analysis:")
                print(f"Detected Condition: {result['disease']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print("\nTop Potato Disease Matches:")
                for i, idx in enumerate(sorted_indices[:3]):
                    if idx < len(self.classes):
                        prob = predictions[0][idx]
                        class_name = self.classes[idx].split('___')[1].replace('_', ' ')
                        print(f"{i+1}. {class_name}: {prob:.2%}")
            else:
                # For uploaded images, keep existing behavior
                predicted_class = sorted_indices[0]
                dataset = PlantDiseaseDataset()
                result = dataset.get_disease_info(predicted_class)
                result['confidence'] = float(predictions[0][predicted_class])
                result['image_path'] = image_path
                result['source_type'] = source_type
                
                # Print debug information
                print(f"\nPrediction for uploaded image:")
                print(f"Selected class: {self.classes[predicted_class]}: {result['confidence']:.2%}")
                print("\nTop 5 predictions:")
                for i, idx in enumerate(sorted_indices[:5]):
                    if idx < len(self.classes):
                        prob = predictions[0][idx]
                        class_name = self.classes[idx]
                        print(f"{i+1}. {class_name}: {prob:.2%}")
            
            # Add warning if confidence is low
            if result['confidence'] < 0.70:
                result['warning'] = 'Low confidence prediction. Please provide a clearer image.'
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                'error': f'Error during prediction: {str(e)}',
                'confidence': 0.0
            }
                
    def get_treatment_recommendation(self, disease):
        """Get treatment recommendations based on disease"""
        if 'healthy' in disease.lower():
            return """Maintenance recommendations:
1. Continue regular watering schedule
2. Maintain proper fertilization
3. Monitor for any signs of disease
4. Practice good garden hygiene
5. Ensure adequate sunlight and airflow"""
        
        treatments = {
            'Apple scab': """Treatment recommendations:
1. Remove and destroy infected leaves and fruit
2. Apply fungicide containing captan or myclobutanil
3. Improve air circulation through proper pruning
4. Apply preventive fungicide in early spring
5. Maintain proper tree spacing""",

            'Late blight': """Treatment recommendations:
1. Remove and destroy infected plants immediately
2. Apply copper-based fungicide
3. Improve air circulation between plants
4. Avoid overhead watering
5. Plant resistant varieties next season
6. Monitor nearby plants for infection""",

            'Early blight': """Treatment recommendations:
1. Remove infected leaves immediately
2. Apply appropriate fungicide
3. Maintain proper plant spacing
4. Water at the base of plants
5. Mulch around plants""",

            'Leaf Mold': """Treatment recommendations:
1. Remove infected leaves
2. Improve air circulation
3. Reduce humidity levels
4. Apply appropriate fungicide
5. Space plants properly""",

            'Septoria leaf spot': """Treatment recommendations:
1. Remove infected leaves
2. Apply fungicide regularly
3. Avoid overhead watering
4. Provide proper spacing
5. Practice crop rotation""",

            'Bacterial spot': """Treatment recommendations:
1. Remove infected plant parts
2. Apply copper-based bactericide
3. Avoid overhead irrigation
4. Improve air circulation
5. Practice crop rotation
6. Use disease-free seeds"""
        }
        
        # Try to find a matching treatment
        for disease_name, treatment in treatments.items():
            if disease_name.lower() in disease.lower():
                return treatment
        
        return """General disease management recommendations:
1. Remove infected plant parts
2. Improve air circulation
3. Avoid overhead watering
4. Apply appropriate fungicide or pesticide
5. Practice crop rotation
6. Consult a local agricultural expert for specific treatment"""
