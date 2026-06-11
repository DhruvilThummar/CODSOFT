import os
import urllib.request
from tqdm import tqdm

# This class helps us show a progress bar when downloading files
# It updates the progress bar as chunks of data are downloaded
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# This function downloads a file from a web link (url) and saves it to a local folder (output_path)
def download_url(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    # Open the progress bar and download the file
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    # Define the folder where we will store the movie datasets
    # We name this folder "Genre_Classification_Dataset"
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Genre_Classification_Dataset")
    os.makedirs(dataset_dir, exist_ok=True) # Create the folder if it does not exist yet
    
    # These are the links to the movie datasets hosted on GitHub
    # train_data.txt: contains movie details with genre labels for training our models
    # test_data.txt: contains movie details without genre labels for prediction
    # test_data_solution.txt: contains movie details with genre labels to test if our model works well
    files = {
        "train_data.txt": "https://raw.githubusercontent.com/Arafath-MSM/Codsoft_MOVIE-GENRE-CLASSIFICATION/main/train_data.txt",
        "test_data.txt": "https://raw.githubusercontent.com/Arafath-MSM/Codsoft_MOVIE-GENRE-CLASSIFICATION/main/test_data.txt",
        "test_data_solution.txt": "https://raw.githubusercontent.com/Arafath-MSM/Codsoft_MOVIE-GENRE-CLASSIFICATION/main/test_data_solution.txt"
    }
    
    # Loop through each file, check if we already have it, and download it if missing
    for filename, url in files.items():
        output_path = os.path.join(dataset_dir, filename)
        if os.path.exists(output_path):
            print(f"{filename} already exists, skipping download.")
        else:
            try:
                download_url(url, output_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    main()
