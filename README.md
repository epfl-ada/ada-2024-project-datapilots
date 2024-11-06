# Cultural taste trends and regional biases in beer preferences around the world

## Project information 
### Abstract:
*A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?*


### Research questions:
*A list of research questions you would like to address during the project.*

### Methods:

### Proposed timeline:

### Organization within the team:
*A list of internal milestones up until project Milestone P3.*

### Questions for TAs (optional):


## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-datapilots.git
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```


## How to use the library

### Project Structure

The directory is organized in the following way:

```
├── data                        <- Project data files (absent from the remote)
│
├── src                         <- Source code
│   ├── data                    <- Data directory
│   │   ├── download_files.py           <- Script to download data from Google Drive
│   │   ├── extract_beer_advocate.py    <- Script to extract downloaded BeerAdvocate data
│   │   └── extract_rate_beer.py        <- Script to extract downloaded RateBeer data 
│   ├── models                  <- Model directory
│   ├── utils                   <- Utility directory
│   ├── scripts                 <- Scripts directory
│   │   └── review_parser.py            <- Script that processes each ratings.txt file by dividing it into parts, parsing each part and
│                                          saving as JSON
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- Notebook containing our analyses
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md

```

### Data Setup Instructions

This project requires two data files: BeerAdvocate.tar.gz and RateBeer.tar.gz. Due to the large size of these files, they are hosted externally on Google Drive and need to be downloaded and extracted locally. The following steps will guide you through the data setup process.

1) Go to the project root directory and run the download_files.py script to download BeerAdvocate.tar.gz and RateBeer.tar.gz to your local machine:
```bash
    python src/data/download_files.py
```
This will create a subfolder called "downloaded_files" in the "data" folder in the root directory, which will contain the downloaded .tar.gz files.


2) After downloading, each .tar.gz file needs to be extracted into a separate folder.

To extract BeerAdvocate.tar.gz, go to the project root directory and run the extract_beer_advocate.py script:
```bash
    python src/data/extract_beer_advocate.py
```
This will extract the contents of BeerAdvocate.tar.gz into a subfolder called BeerAdvocate in the "data" folder in the root directory. This script also extracts the ratings.txt.gz file in the subfolder.


To extract RateBeer.tar.gz, run the extract_rate_beer.py script:
```bash
    python src/data/extract_rate_beer.py
```
This will extract the contents of RateBeer.tar.gz into a subfolder called RateBeer in the "data" folder in the root directory. This script also extracts the ratings.txt.gz file in the subfolder.


3) ratings.txt files are extremely large, and trying to load them directly into DataFrames leads to kernel freezes. In order to circumvent this problem, we wrote a script (review_parser.py, located in src/scripts), which processes each rating file by dividing it into parts, parsing each part, and saving as JSON. In the results.ipynb notebook, we then load the different JSON files into DataFrames, that we concatenate. In order to perform this operation, go to the project root directory and run the review_parser.py script:
```bash
    python src/scripts/review_parser.py
```
This will create 3 JSON files in both the BeerAdvocate and the RateBeer subfolders.
