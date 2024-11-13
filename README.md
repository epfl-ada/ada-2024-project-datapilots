# Selective sips: cultural and bias trends in beer preferences across countries

## Project information 
### Abstract:
Taste preferences for food and drinks often go beyond the intrinsic characteristics of the items themselves and are in reality shaped by various external influences. Cultural differences are a prime example: while highly spiced dishes are popular in many South Asian countries, milder flavors are often preferred in Western Europe. This project examines similar external factors that shape beer preferences, aiming to uncover what truly drives an individual’s taste in beers. We analyze how different beer characteristics are appreciated across selected countries and assess whether the origin of a beer biases the ratings it receives. Additionally, we investigate how seasonal variations and user experience impact the enjoyment of specific beer categories. By identifying these “external” influences, we hope to help beer enthusiasts better understand their preferences and make choices based more on intrinsic qualities, ultimately improving their sensory experience and enjoyment of beer.

### Research questions:
**1) Cultural influence on beer preferences**

a) How do beer style preferences differ between countries, and are these regional preferences stable over time?

b) Does the significance of specific beer attributes in determining one’s liking of a given beer vary by country?


**2) Location-related biases in ratings**

a) Are users from certain countries more generous or more critical in their ratings compared to users from other countries?

b) Do users show a bias by rating domestic beers higher than foreign ones?


**3) Other biases in ratings**

a) Do seasonal changes affect how different beer styles are rated?

b) Do users become more critical with experience?


### Dataset
The dataset for this analysis comprises beer reviews collected from two popular beer rating platforms, BeerAdvocate and RateBeer, covering a period from 2001 to 2017. For each website, the dataset includes metadata on reviewers, beers, and breweries, along with detailed user reviews. In total, there are records of over 500,000 unique beers produced by breweries in more than 200 countries. Among the most frequently reviewed beer styles are American IPA and India Pale Ale. The dataset also includes approximately 200,000 users from over 200 countries, though the distribution of users and breweries is heavily skewed: the vast majority are located in the United States on both platforms. Overall, the dataset contains over 8 million reviews from BeerAdvocate and 7 million from RateBeer. For the parts of our analysis that involve country comparisons, we excluded reviews from countries with fewer than 50 reviewers to ensure that the data is representative at a national level.


### Methods:
**Cultural influence on beer preferences**

*Beer style preferences*

We employ clustering techniques to identify the most popular beer styles in each country. Additionally, we  use the time information contained in beer reviews to determine whether regional beer style preferences remain stable over time—supporting the hypothesis that they are influenced by culture—or if they fluctuate, suggesting other contributing factors.

*Importance of specific beer attributes*

To assess the overall importance of various beer attributes in determining the final rating, we use linear regression with attribute ratings as features and the final rating as the outcome variable across all countries combined. We then compare the coefficients of each attribute to evaluate their relative influence. To understand how the importance of these attributes varies between countries, we repeat this analysis separately for each country and examine the distribution of attribute coefficients across different countries.


**Location-related biases in ratings**

*Cultural biases*

To see whether users from certain countries are more generous or more critical in their ratings compared to users from other countries, we calculate the final rating for each country and use statistical tests to determine if the final rating is the same for each country.

*Beer origin bias*

To assess whether users rate domestic beers higher than foreign ones, we compare the final ratings of domestic versus foreign beers and use statistical tests to determine if the difference is significant. To further investigate this potential bias, we analyze whether the final rating of a beer correlates with the number of reviewers from the beer's country of origin by employing scatter plots, Pearson’s correlation coefficient, and linear regression. We repeat this analysis focusing only on reviews from beer enthusiasts—users who have written a substantial number of reviews—who might prioritize intrinsic qualities of the beers over external factors such as location.


**Other biases**

*Seasonal biases*

To examine how seasonal changes influence the ratings of different beer styles, we use the time information contained in ratings to identify the season during which each rating was posted, taking into account the location of the user (Northern hemisphere, Southern hemisphere or equatorial area) to accurately determine the season. For simplicity, we only perform the analysis on users from the 10 countries with the highest number of reviews. We then group the ratings by season and calculate the average final rating within each group to see if some seasons are associated with higher ratings overall. We then refine the analysis by calculating the average final rating for each beer style within each group, and compare these averages across different seasons to identify any notable variations.

*Experience bias*

To analyze how users’ rating tendencies evolve with experience, we focus on users who have posted a substantial number of reviews, based on a chosen threshold. For each user, we sort their reviews chronologically and assign an "experience level" to each rating based on the number of reviews they had posted up to that point. These levels are predefined and consistent across all users: new reviewer (first n reviews), amateur (from the n+1th to the oth review), and expert (from the o+1th review onward). We then calculate and compare the average final rating for each experience level across all users. If a discernible trend emerges, we perform a paired t-test (comparing early vs. later reviews by the same user) to determine whether any observed increase or decrease in ratings is statistically significant.


### Proposed timeline:
**Week 7:** projet choice

**Week 8:** data loading and cleaning

**Week 9:** verification of project feasibility (verification that there is enough data for the different tasks), writing of the README

**Weeks 12 and 13:** implementation of the analyses described in "Methods"

**Week 14:** writing of the data story


### Organization within the team:
**Data loading:** Athénaïs

**Data cleaning:** Orkun and Athénaïs

**Beer style preferences:** Orkun

**Importance of specific beer attributes:** Samet

**Cultural biases & Beer origin bias:** Melis

**Seasonal biases:** Begüm

**Experience bias:** Athénaïs

**Data story:** everyone (everyone will add complete the data story with elements related to the part they worked on)


### Questions for TAs:


## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-datapilots.git
cd <project repo>

# [OPTIONAL] create conda environment
conda create -y -n <env_name> python=3.11 scipy pandas numpy matplotlib=3.7.2
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
