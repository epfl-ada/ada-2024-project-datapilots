import tarfile
import os

# Get the path to the parent directory of the 'src' folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the paths for the download directory and the BeerAdvocate extraction directory
data_dir = os.path.join(base_dir, 'data')
download_dir = os.path.join(data_dir, 'downloaded_files')
beer_advocate_extract_dir = os.path.join(data_dir, "BeerAdvocate")

# Ensure the extraction directory exists
os.makedirs(beer_advocate_extract_dir, exist_ok=True)

# Path to the downloaded BeerAdvocate.tar.gz file
beer_advocate_download_path = os.path.join(download_dir, "BeerAdvocate.tar.gz")

# Extract BeerAdvocate.tar.gz into 'BeerAdvocate' folder
print(f"Extracting BeerAdvocate.tar.gz to {beer_advocate_extract_dir}...")
with tarfile.open(beer_advocate_download_path, "r:gz") as tar:
    tar.extractall(path=beer_advocate_extract_dir)
print(f"BeerAdvocate.tar.gz extracted to {beer_advocate_extract_dir}")
