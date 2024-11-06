import tarfile
import gzip
import shutil
import os

# Get the path to the project root directory (move up two levels from src/data)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define the paths for the data directory and the download directory
data_dir = os.path.join(base_dir, 'data')
download_dir = os.path.join(data_dir, 'downloaded_files')

# Define the extraction directory at the root data level
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

# Path to the ratings.txt.gz file inside the BeerAdvocate folder
ratings_gz_path = os.path.join(beer_advocate_extract_dir, "ratings.txt.gz")
ratings_txt_path = os.path.join(beer_advocate_extract_dir, "ratings.txt")

# Check if ratings.txt.gz exists and extract it
if os.path.exists(ratings_gz_path):
    print(f"Extracting {ratings_gz_path}...")
    with gzip.open(ratings_gz_path, 'rb') as f_in:
        with open(ratings_txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"{ratings_gz_path} extracted to {ratings_txt_path}")
else:
    print(f"File {ratings_gz_path} does not exist.")
