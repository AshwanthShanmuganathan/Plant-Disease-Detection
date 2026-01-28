import os
import kaggle
import zipfile
import shutil
from tqdm import tqdm

def setup_kaggle_credentials():
    """Setup Kaggle credentials if not already set up"""
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        print("\nKaggle API credentials not found. Please follow these steps:")
        print("1. Go to www.kaggle.com and create an account if you haven't already")
        print("2. Go to your Kaggle account settings (https://www.kaggle.com/settings)")
        print("3. Scroll to 'API' section and click 'Create New API Token'")
        print("4. This will download a kaggle.json file")
        print("5. Place the downloaded kaggle.json file in this directory\n")
        
        while True:
            kaggle_json = input("Have you downloaded and placed the kaggle.json file here? (yes/no): ").lower()
            if kaggle_json == 'yes':
                # Create .kaggle directory if it doesn't exist
                os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
                
                # Move kaggle.json to the right location
                if os.path.exists('kaggle.json'):
                    shutil.move('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
                    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
                    print("Credentials set up successfully!")
                    break
                else:
                    print("kaggle.json not found in the current directory. Please try again.")
            elif kaggle_json == 'no':
                print("Please download the kaggle.json file and try again.")
                return False
    return True

def download_dataset():
    """Download the PlantVillage dataset from Kaggle"""
    print("Starting dataset download...")
    
    # Create necessary directories
    os.makedirs('dataset', exist_ok=True)
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            'abdallahalidev/plantvillage-dataset',
            path='dataset',
            quiet=False
        )
        
        print("\nExtracting dataset...")
        with zipfile.ZipFile('dataset/plantvillage-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('dataset')
        
        # Organize into train and validation directories
        organize_dataset()
        
        # Clean up
        os.remove('dataset/plantvillage-dataset.zip')
        print("\nDataset downloaded and organized successfully!")
        print("\nDataset Information:")
        print("- Location: dataset/train and dataset/val")
        print("- Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("- Contains multiple plant species and their diseases")
        
    except Exception as e:
        print(f"\nError downloading dataset: {str(e)}")
        return False
    
    return True

def organize_dataset():
    """Organize the downloaded dataset into train and validation splits"""
    source_dir = 'dataset/PlantVillage'
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all categories
    categories = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))]
    
    print("\nOrganizing dataset into train (80%) and validation (20%) sets...")
    for category in tqdm(categories):
        # Create category directories
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # Get all images
        images = [img for img in os.listdir(os.path.join(source_dir, category))
                 if img.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'))]
        
        # Split into train and validation
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy training images
        for img in train_images:
            src = os.path.join(source_dir, category, img)
            dst = os.path.join(train_dir, category, img)
            shutil.copy2(src, dst)
        
        # Copy validation images
        for img in val_images:
            src = os.path.join(source_dir, category, img)
            dst = os.path.join(val_dir, category, img)
            shutil.copy2(src, dst)

def main():
    print("PlantVillage Dataset Downloader")
    print("===============================")
    print("This will download the complete PlantVillage dataset from Kaggle.")
    print("Dataset source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("Size: ~2GB (compressed)")
    
    if not setup_kaggle_credentials():
        return
    
    download_dataset()

if __name__ == "__main__":
    main()
