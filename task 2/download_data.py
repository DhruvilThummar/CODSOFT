import os
import urllib.request
import zipfile
from tqdm import tqdm

# Progress bar class for tracking the download
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "Credit_Card_Fraud_Dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    zip_path = os.path.join(dataset_dir, "creditcard.csv.zip")
    csv_path = os.path.join(dataset_dir, "creditcard.csv")
    
    # URL for downloading the zipped dataset
    url = "https://raw.githubusercontent.com/stat432/credit-analysis/main/data-raw/creditcard.csv.zip"
    
    # Download file if it doesn't exist already
    if os.path.exists(csv_path):
        print(f"Dataset already extracted at: {csv_path}")
        return
        
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="creditcard.csv.zip") as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return
            
    # Extract zip file
    print("Extracting creditcard.csv.zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"Extraction complete. CSV saved at: {csv_path}")
        # Remove the zip file to clean up space
        os.remove(zip_path)
        print("Cleaned up zip archive.")
    except Exception as e:
        print(f"Failed to extract zip file: {e}")

if __name__ == "__main__":
    download_and_extract()
