import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def prepare_dataset():
    """Prepare and organize the complete PlantVillage dataset"""
    # Main source directory containing all image types
    source_dir = r"C:\Users\Yakshini\Downloads\archive\plantvillage dataset"
    
    # Get all dataset types (color, grayscale, segmented, etc.)
    dataset_types = [d for d in os.listdir(source_dir) 
                    if os.path.isdir(os.path.join(source_dir, d))]
    
    # Create source_dirs dictionary dynamically
    source_dirs = {
        dataset_type: os.path.join(source_dir, dataset_type)
        for dataset_type in dataset_types
    }
    
    # Destination directories
    base_dir = os.path.join(os.getcwd(), 'dataset')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    # Create destination directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("\nPreparing PlantVillage dataset with multiple image types...")
    
    for img_type, source_dir in source_dirs.items():
        print(f"\nProcessing {img_type} images from: {source_dir}")
        
        # Get all categories
        if os.path.exists(source_dir):
            categories = [d for d in os.listdir(source_dir) 
                        if os.path.isdir(os.path.join(source_dir, d))]
            
            print(f"Found {len(categories)} categories in {img_type} dataset")
            
            # Process each category
            for category in categories:
                print(f"\nProcessing {category}")
                
                # Create category directories
                train_category_dir = os.path.join(train_dir, category)
                val_category_dir = os.path.join(val_dir, category)
                os.makedirs(train_category_dir, exist_ok=True)
                os.makedirs(val_category_dir, exist_ok=True)
                
                # Get all images in the category
                source_category_dir = os.path.join(source_dir, category)
                images = [f for f in os.listdir(source_category_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Split into train and validation
                train_images, val_images = train_test_split(
                    images, test_size=0.2, random_state=42)
                
                # Copy training images
                for img in tqdm(train_images, desc=f"Copying {img_type} training images"):
                    src = os.path.join(source_category_dir, img)
                    # Add image type prefix to avoid name conflicts
                    dst = os.path.join(train_category_dir, 
                                     f"{img_type}_{category}_{img}")
                    shutil.copy2(src, dst)
                
                # Copy validation images
                for img in tqdm(val_images, desc=f"Copying {img_type} validation images"):
                    src = os.path.join(source_category_dir, img)
                    # Add image type prefix to avoid name conflicts
                    dst = os.path.join(val_category_dir,
                                     f"{img_type}_{category}_{img}")
                    shutil.copy2(src, dst)
                
                print(f"  {category}:")
                print(f"    Training images: {len(train_images)}")
                print(f"    Validation images: {len(val_images)}")
        else:
            print(f"Warning: {img_type} source directory not found at {source_dir}")

    print("\nDataset preparation completed!")
    print_dataset_statistics(train_dir, val_dir)

def print_dataset_statistics(train_dir, val_dir):
    """Print statistics about the prepared dataset"""
    print("\nDataset Statistics:")
    total_train = 0
    total_val = 0
    
    for category in sorted(os.listdir(train_dir)):
        train_count = len(os.listdir(os.path.join(train_dir, category)))
        val_count = len(os.listdir(os.path.join(val_dir, category)))
        total_train += train_count
        total_val += val_count
        print(f"\n{category}:")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")
    
    print(f"\nTotal:")
    print(f"  Total training images: {total_train}")
    print(f"  Total validation images: {total_val}")
    print(f"  Total images: {total_train + total_val}")

if __name__ == "__main__":
    prepare_dataset()
