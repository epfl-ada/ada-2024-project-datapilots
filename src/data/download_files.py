import gdown
import os

# Google Drive file IDs for each .tar.gz file
beer_advocate_file_id = "1IqcAJtYrDB1j40rBY5M-PGp6KNX-E3xq"  
rate_beer_file_id = "1vt-CTz6Ni8fPTIkHehW9Mm0RPMpvkH3a"          

# Define the path to the project root directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the main data directory and download directory at the project root level
data_dir = os.path.join(base_dir, 'data')
download_dir = os.path.join(data_dir, 'downloaded_files')

# Ensure the download directory exists at the desired location
os.makedirs(download_dir, exist_ok=True)

# Download BeerAdvocate.tar.gz
beer_advocate_download_path = os.path.join(download_dir, "BeerAdvocate.tar.gz")
print("Downloading BeerAdvocate.tar.gz from Google Drive...")
gdown.download(f"https://drive.google.com/uc?id={beer_advocate_file_id}", beer_advocate_download_path, quiet=False)
print(f"Downloaded BeerAdvocate.tar.gz to {beer_advocate_download_path}")

# Download RateBeer.tar.gz
rate_beer_download_path = os.path.join(download_dir, "RateBeer.tar.gz")
print("Downloading RateBeer.tar.gz from Google Drive...")
gdown.download(f"https://drive.google.com/uc?id={rate_beer_file_id}", rate_beer_download_path, quiet=False)
print(f"Downloaded RateBeer.tar.gz to {rate_beer_download_path}")

print("Files have been downloaded.")

