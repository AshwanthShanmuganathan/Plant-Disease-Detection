import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class PlantDiseaseDataset:
    def __init__(self):
        self.class_names = self._load_class_names()
        self.scientific_names = {
            # Fruits
            'Apple': 'Malus domestica',
            'Apricot': 'Prunus armeniaca',
            'Avocado': 'Persea americana',
            'Banana': 'Musa acuminata',
            'Blackberry': 'Rubus fruticosus',
            'Blueberry': 'Vaccinium corymbosum',
            'Cherry': 'Prunus avium',
            'Fig': 'Ficus carica',
            'Grape': 'Vitis vinifera',
            'Guava': 'Psidium guajava',
            'Kiwi': 'Actinidia deliciosa',
            'Lemon': 'Citrus limon',
            'Lime': 'Citrus aurantifolia',
            'Mango': 'Mangifera indica',
            'Orange': 'Citrus × sinensis',
            'Papaya': 'Carica papaya',
            'Peach': 'Prunus persica',
            'Pear': 'Pyrus communis',
            'Pineapple': 'Ananas comosus',
            'Plum': 'Prunus domestica',
            'Pomegranate': 'Punica granatum',
            'Raspberry': 'Rubus idaeus',
            'Strawberry': 'Fragaria × ananassa',

            # Vegetables
            'Asparagus': 'Asparagus officinalis',
            'Beetroot': 'Beta vulgaris',
            'Broccoli': 'Brassica oleracea var. italica',
            'Cabbage': 'Brassica oleracea var. capitata',
            'Carrot': 'Daucus carota',
            'Cauliflower': 'Brassica oleracea var. botrytis',
            'Celery': 'Apium graveolens',
            'Cucumber': 'Cucumis sativus',
            'Eggplant': 'Solanum melongena',
            'Garlic': 'Allium sativum',
            'Ginger': 'Zingiber officinale',
            'Lettuce': 'Lactuca sativa',
            'Onion': 'Allium cepa',
            'Pepper bell': 'Capsicum annuum',
            'Potato': 'Solanum tuberosum',
            'Pumpkin': 'Cucurbita pepo',
            'Radish': 'Raphanus sativus',
            'Spinach': 'Spinacia oleracea',
            'Squash': 'Cucurbita pepo',
            'Sweet potato': 'Ipomoea batatas',
            'Tomato': 'Solanum lycopersicum',
            'Turnip': 'Brassica rapa',
            'Zucchini': 'Cucurbita pepo var. cylindrica',

            # Grains and Legumes
            'Barley': 'Hordeum vulgare',
            'Corn': 'Zea mays',
            'Chickpea': 'Cicer arietinum',
            'Lentil': 'Lens culinaris',
            'Millet': 'Pennisetum glaucum',
            'Oats': 'Avena sativa',
            'Peanut': 'Arachis hypogaea',
            'Rice': 'Oryza sativa',
            'Rye': 'Secale cereale',
            'Soybean': 'Glycine max',
            'Wheat': 'Triticum aestivum',

            # Herbs and Spices
            'Basil': 'Ocimum basilicum',
            'Cilantro': 'Coriandrum sativum',
            'Mint': 'Mentha',
            'Oregano': 'Origanum vulgare',
            'Parsley': 'Petroselinum crispum',
            'Rosemary': 'Rosmarinus officinalis',
            'Sage': 'Salvia officinalis',
            'Thyme': 'Thymus vulgaris',

            # Other Commercial Crops
            'Coffee': 'Coffea arabica',
            'Cotton': 'Gossypium hirsutum',
            'Sugarcane': 'Saccharum officinarum',
            'Tea': 'Camellia sinensis',
            'Tobacco': 'Nicotiana tabacum'
        }
        self.treatment_recommendations = self._load_treatment_recommendations()

    def _load_class_names(self):
        """Dynamically load class names from the dataset"""
        return {
            0: 'Tomato___healthy',
            1: 'Tomato___Late_blight',
            2: 'Tomato___Early_blight',
            3: 'Tomato___Leaf_Mold',
            4: 'Tomato___Septoria_leaf_spot',
            5: 'Potato___healthy',
            6: 'Potato___Early_blight',
            7: 'Potato___Late_blight',
            8: 'Apple___healthy',
            9: 'Apple___Apple_scab',
            10: 'Apple___Black_rot'
        }

    def _load_treatment_recommendations(self):
        """Load treatment recommendations for all diseases"""
        base_treatments = {
            "healthy": {
                "treatment": "No treatment needed - plant is healthy",
                "prevention": "1. Regular monitoring\n2. Proper watering\n3. Balanced fertilization\n4. Good air circulation\n5. Adequate sunlight"
            },
            "Apple_scab": {
                "treatment": "1. Remove and destroy infected leaves\n2. Apply fungicides with active ingredients like captan or myclobutanil\n3. Maintain good air circulation by proper pruning",
                "prevention": "1. Plant resistant varieties\n2. Clean up fallen leaves in autumn\n3. Improve air circulation"
            },
            "Black_rot": {
                "treatment": "1. Remove infected fruit and cankers\n2. Apply copper-based fungicides\n3. Prune during dry weather",
                "prevention": "1. Regular pruning\n2. Remove mummified fruit\n3. Maintain proper spacing between plants"
            },
            "Bacterial_spot": {
                "treatment": "1. Remove infected plant parts\n2. Apply copper-based bactericides\n3. Improve air circulation",
                "prevention": "1. Use disease-free seeds\n2. Rotate crops\n3. Avoid overhead irrigation"
            },
            "Early_blight": {
                "treatment": "1. Remove infected leaves\n2. Apply fungicide containing chlorothalonil\n3. Maintain proper spacing",
                "prevention": "1. Mulch around plants\n2. Water at base of plant\n3. Practice crop rotation"
            },
            "Late_blight": {
                "treatment": "1. Remove infected plants immediately\n2. Apply copper-based fungicide\n3. Destroy all infected material",
                "prevention": "1. Plant resistant varieties\n2. Improve drainage\n3. Monitor weather conditions"
            },
            "Leaf_Mold": {
                "treatment": "1. Remove infected leaves\n2. Apply fungicide\n3. Reduce humidity",
                "prevention": "1. Improve air circulation\n2. Avoid leaf wetness\n3. Space plants properly"
            },
            "Powdery_mildew": {
                "treatment": "1. Apply sulfur-based fungicide\n2. Remove infected plant parts\n3. Increase air circulation",
                "prevention": "1. Plant resistant varieties\n2. Space plants properly\n3. Avoid overhead watering"
            },
            "Black_spot": {
                "treatment": "1. Remove affected leaves immediately\n2. Apply fungicide specifically for black spot\n3. Improve air circulation",
                "prevention": "1. Space plants properly\n2. Water at base of plant\n3. Clean up fallen leaves"
            },
            "Rust": {
                "treatment": "1. Remove infected plant parts\n2. Apply appropriate fungicide\n3. Improve air circulation",
                "prevention": "1. Avoid overhead watering\n2. Space plants properly\n3. Keep leaves dry"
            },
            "Leaf_spot": {
                "treatment": "1. Remove infected leaves\n2. Apply fungicide\n3. Improve air circulation",
                "prevention": "1. Avoid overhead watering\n2. Space plants properly\n3. Clean garden tools"
            },
            "Botrytis_blight": {
                "treatment": "1. Remove infected flowers and leaves\n2. Apply appropriate fungicide\n3. Reduce humidity",
                "prevention": "1. Improve air circulation\n2. Avoid overhead watering\n3. Remove dead plant material"
            },
            "Black_scurf": {
                "treatment": "1. Remove infected tubers\n2. Apply appropriate fungicide to soil\n3. Harvest promptly when mature",
                "prevention": "1. Use certified seed potatoes\n2. Practice crop rotation\n3. Improve soil drainage"
            },
            "Common_scab": {
                "treatment": "1. Maintain soil pH below 5.5\n2. Increase organic matter in soil\n3. Ensure consistent soil moisture",
                "prevention": "1. Use resistant varieties\n2. Avoid adding lime to soil\n3. Practice crop rotation"
            },
            "Ring_rot": {
                "treatment": "1. Remove and destroy infected plants\n2. Clean and disinfect all equipment\n3. Implement strict sanitation",
                "prevention": "1. Use certified disease-free seed\n2. Clean tools between uses\n3. Practice crop rotation"
            },
            "Soft_rot": {
                "treatment": "1. Remove infected tubers\n2. Improve storage conditions\n3. Maintain proper ventilation",
                "prevention": "1. Avoid wounding during harvest\n2. Store at proper temperature\n3. Ensure good air circulation"
            },
            "Leaf_roll_virus": {
                "treatment": "1. Remove infected plants\n2. Control aphid populations\n3. Use virus-free seed potatoes",
                "prevention": "1. Plant certified seed\n2. Control insect vectors\n3. Isolate from infected fields"
            },
            "Mosaic_virus": {
                "treatment": "1. Remove and destroy infected plants\n2. Control aphid populations\n3. Use virus-resistant varieties",
                "prevention": "1. Use certified seed potatoes\n2. Control weed hosts\n3. Monitor for aphids"
            },
            "Root_knot": {
                "treatment": "1. Remove heavily infected plants\n2. Apply nematicides if severe\n3. Improve soil health",
                "prevention": "1. Rotate with non-host crops\n2. Use resistant varieties\n3. Add organic matter to soil"
            },
            "Blackleg": {
                "treatment": "1. Remove infected plants and tubers\n2. Improve drainage\n3. Apply copper-based bactericides",
                "prevention": "1. Use certified seed potatoes\n2. Avoid overwatering\n3. Practice crop rotation"
            }
        }

        # Add flower-specific scientific names
        self.scientific_names.update({
            'Rose': 'Rosa species',
            'Sunflower': 'Helianthus annuus',
            'Marigold': 'Tagetes species',
            'Jasmine': 'Jasminum species',
            'Hibiscus': 'Hibiscus rosa-sinensis'
        })

        # Add generic treatment for any disease not specifically listed
        generic_treatment = {
            "treatment": "1. Remove infected plant parts\n2. Apply appropriate fungicide\n3. Improve growing conditions",
            "prevention": "1. Monitor regularly\n2. Maintain plant hygiene\n3. Ensure proper spacing"
        }

        return base_treatments, generic_treatment

    def create_model(self, num_classes=38):
        """Create a ResNet50V2 model pre-trained on ImageNet"""
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def get_disease_info(self, class_idx):
        """Get disease information and treatment recommendations"""
        class_name = self.class_names.get(class_idx, "Unknown")
        
        # Extract plant name and disease
        parts = class_name.split("___")
        plant = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        
        # Get scientific name
        scientific_name = self.scientific_names.get(plant, "Scientific name not available")
        
        # Get treatment recommendations
        disease_key = parts[1] if len(parts) > 1 else "healthy"
        base_treatments, generic_treatment = self.treatment_recommendations
        treatment_info = base_treatments.get(disease_key, generic_treatment)
        
        return {
            "plant_name": plant,
            "disease": disease,
            "scientific_name": scientific_name,
            "treatment": treatment_info["treatment"],
            "prevention": treatment_info["prevention"]
        }
