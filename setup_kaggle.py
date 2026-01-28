import os
import json
import getpass

def setup_kaggle_credentials():
    """Set up Kaggle API credentials"""
    print("Setting up Kaggle credentials...")
    print("Please visit https://www.kaggle.com/account and create an API token if you haven't already")
    
    username = input("Enter your Kaggle username: ")
    key = getpass.getpass("Enter your Kaggle API key: ")
    
    # Create kaggle directory if it doesn't exist
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create kaggle.json
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_json, 'w') as f:
        json.dump({
            "username": username,
            "key": key
        }, f)
    
    # Set appropriate permissions
    os.chmod(kaggle_json, 0o600)
    
    print("\nKaggle credentials have been set up!")
    print("You can now run download_dataset.py to download the plant disease dataset")

if __name__ == "__main__":
    setup_kaggle_credentials()
