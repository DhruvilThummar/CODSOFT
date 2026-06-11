import os
import urllib.request
from tqdm import tqdm

# Progress bar class for tracking the download
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dataset():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "SMS_Spam_Dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    csv_path = os.path.join(dataset_dir, "spam.csv")
    
    # URL for downloading the SMS Spam Collection dataset from mohitgupta-omg repository
    url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    
    # Download file if it doesn't exist already
    if os.path.exists(csv_path):
        print(f"Dataset already exists at: {csv_path}")
        return
        
    print(f"Downloading dataset from {url}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="spam.csv") as t:
            urllib.request.urlretrieve(url, filename=csv_path, reporthook=t.update_to)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)

if __name__ == "__main__":
    download_dataset()
