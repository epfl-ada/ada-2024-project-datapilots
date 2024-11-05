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





## Information about installation, usage and structure
## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.


### Data Setup Instructions

This project requires two data files: BeerAdvocate.tar.gz and RateBeer.tar.gz. Due to the large size of these files, they are hosted externally on Google Drive and need to be downloaded and extracted locally. The following steps will guide you through the data setup process.

1) Run the download_files.py script to download BeerAdvocate.tar.gz and RateBeer.tar.gz to your local machine.
    python src/data/download_files.py

2) After downloading, each .tar.gz file needs to be extracted into a separate folder.
To extract BeerAdvocate.tar.gz, run the extract_beer_advocate.py script:
    python src/data/extract_beer_advocate.py

This will extract the contents of BeerAdvocate.tar.gz into a folder called BeerAdvocate

To extract RateBeer.tar.gz, run the extract_rate_beer.py script:
    python src/data/extract_rate_beer.py

This will extract the contents of RateBeer.tar.gz into a folder called RateBeer.


## Project Structure

After completing these steps, your directory structure should look like this:

```
├── src                         <- Source code
│   ├── data                            <- Project data files and data loader
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```