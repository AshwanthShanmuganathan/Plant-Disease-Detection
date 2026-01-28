import os
import numpy as np
from PIL import Image, ImageDraw

def create_synthetic_image(category, index, size=(224, 224)):
    """Create a synthetic plant image for testing"""
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    if 'healthy' in category.lower():
        # Draw a healthy leaf (green)
        color = (34, 139, 34)  # Forest green
        draw.ellipse([50, 50, 174, 174], fill=color)
    elif 'scab' in category.lower():
        # Draw a leaf with spots (brown spots on green)
        color = (34, 139, 34)  # Forest green
        draw.ellipse([50, 50, 174, 174], fill=color)
        # Add brown spots
        for _ in range(5):
            x = np.random.randint(60, 164)
            y = np.random.randint(60, 164)
            draw.ellipse([x, y, x+20, y+20], fill=(139, 69, 19))
    elif 'blight' in category.lower():
        # Draw a blighted leaf (yellow-brown)
        color = (205, 133, 63)  # Peru (brownish)
        draw.ellipse([50, 50, 174, 174], fill=color)
        # Add dark spots
        for _ in range(8):
            x = np.random.randint(60, 164)
            y = np.random.randint(60, 164)
            draw.ellipse([x, y, x+15, y+15], fill=(101, 67, 33))
    
    return image

def create_synthetic_dataset(num_images_per_category=50):
    """Create a synthetic dataset for testing"""
    categories = [
        'Apple___healthy',
        'Apple___Apple_scab',
        'Tomato___healthy',
        'Tomato___Late_blight'
    ]
    
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    # Create directories
    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    
    print("Creating synthetic dataset...")
    for category in categories:
        print(f"\nGenerating images for {category}")
        for i in range(num_images_per_category):
            # Create image
            image = create_synthetic_image(category, i)
            
            # Save to train or validation (80-20 split)
            if i < int(0.8 * num_images_per_category):
                save_dir = os.path.join(train_dir, category)
            else:
                save_dir = os.path.join(val_dir, category)
            
            image.save(os.path.join(save_dir, f"{category}_{i}.jpg"))
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} images")

def main():
    print("Creating synthetic dataset for testing...")
    create_synthetic_dataset()
    print("\nSynthetic dataset creation completed!")
    
    # Print statistics
    train_dir = os.path.join(os.getcwd(), 'dataset', 'train')
    val_dir = os.path.join(os.getcwd(), 'dataset', 'val')
    
    print("\nDataset Statistics:")
    for category in os.listdir(train_dir):
        train_count = len(os.listdir(os.path.join(train_dir, category)))
        val_count = len(os.listdir(os.path.join(val_dir, category)))
        print(f"\n{category}:")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")

if __name__ == "__main__":
    main()
