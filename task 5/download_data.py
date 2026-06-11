import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, desc):
    print(f"Downloading {url}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download {desc}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Download and slice Shakespeare corpus
    text_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    temp_text_path = os.path.join(current_dir, "temp_input.txt")
    final_text_path = os.path.join(current_dir, "handwriting_transcripts.txt")
    
    if os.path.exists(final_text_path):
        print(f"Transcripts already exist at: {final_text_path}")
    else:
        download_file(text_url, temp_text_path, "input.txt")
        
        # Read and save the first 100,000 characters (~100KB)
        print("Slicing dataset to 100KB for CPU-friendly training...")
        with open(temp_text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        subset_text = full_text[:100000]
        
        with open(final_text_path, "w", encoding="utf-8") as f:
            f.write(subset_text)
            
        # Clean up temp file
        os.remove(temp_text_path)
        print(f"Saved sliced text transcripts to: {final_text_path}")
        
    # 2. Download Caveat-Regular.ttf font
    font_url = "https://raw.githubusercontent.com/googlefonts/caveat/main/fonts/ttf/Caveat-Regular.ttf"
    font_path = os.path.join(current_dir, "Caveat-Regular.ttf")
    
    if os.path.exists(font_path):
        print(f"Font already exists at: {font_path}")
    else:
        download_file(font_url, font_path, "Caveat-Regular.ttf")
        print(f"Saved handwriting font to: {font_path}")

if __name__ == "__main__":
    main()
