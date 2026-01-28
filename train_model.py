import tensorflow as tf
import os
import numpy as np
from model import PlantDiseaseModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def get_available_classes():
    """Get all available classes from the dataset"""
    train_dir = 'dataset/train'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}. Please prepare the dataset first.")
    
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        raise ValueError("No class directories found in the training directory")
    
    return classes

def create_data_generators():
    """Create data generators for the dataset"""
    # Get available classes
    classes = get_available_classes()
    print(f"\nFound {len(classes)} classes:")
    for cls in classes:
        print(f"- {cls}")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        classes=classes
    )

    print("\nLoading validation data...")
    validation_generator = val_datagen.flow_from_directory(
        'dataset/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        classes=classes
    )

    return train_generator, validation_generator

def train_model():
    """Train the model with the complete PlantVillage dataset"""
    # Get total number of classes from the dataset
    train_dir = os.path.join(os.getcwd(), 'dataset', 'train')
    num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Training model with {num_classes} classes...")
    print("Initializing model training...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Get available classes for model initialization
    classes = get_available_classes()
    
    # Initialize model with correct number of classes
    model = PlantDiseaseModel(num_classes=len(classes))
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Print dataset information
    print(f"\nTraining with {train_generator.samples} training images")
    print(f"Validating with {validation_generator.samples} validation images")
    print(f"Number of classes: {len(train_generator.class_indices)}")
    
    # Print class mapping
    print("\nClass mapping:")
    for class_name, index in train_generator.class_indices.items():
        print(f"  {index}: {class_name}")
    
    # Callbacks for better training
    callbacks = [
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    history = model.model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.model.save('models/final_model.h5')
    
    return history

def plot_training_history(history):
    """Plot the training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history plot saved as 'training_history.png'")

def main():
    print("Starting plant disease detection model training...")
    
    try:
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU acceleration enabled")
        else:
            print("No GPU found. Using CPU for training")
    except Exception as e:
        print(f"GPU configuration error: {e}")
        print("Continuing with default settings...")
    
    # Train the model
    try:
        history = train_model()
        plot_training_history(history)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main()
