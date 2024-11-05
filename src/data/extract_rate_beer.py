import tarfile
import os

# Get the path to the parent directory of the 'src' folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the paths for the download directory and the RateBeer extraction directory
data_dir = os.path.join(base_dir, 'data')
download_dir = os.path.join(data_dir, 'downloaded_files')
rate_beer_extract_dir = os.path.join(data_dir, "RateBeer")

# Ensure the extraction directory exists
os.makedirs(rate_beer_extract_dir, exist_ok=True)

# Path to the downloaded RateBeer.tar.gz file
rate_beer_download_path = os.path.join(download_dir, "RateBeer.tar.gz")

# Extract RateBeer.tar.gz into 'RateBeer' folder
print(f"Extracting RateBeer.tar.gz to {rate_beer_extract_dir}...")
with tarfile.open(rate_beer_download_path, "r:gz") as tar:
    tar.extractall(path=rate_beer_extract_dir)
print(f"RateBeer.tar.gz extracted to {rate_beer_extract_dir}")
