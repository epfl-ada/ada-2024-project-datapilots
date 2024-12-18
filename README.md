# Selective sips: cultural and bias trends in beer preferences across countries

**Link to our data story:** https://athaigueperse.github.io/ada-template-website/

## Project information 
### Abstract:
Taste preferences for food and drinks often go beyond the intrinsic characteristics of the items themselves and are in reality shaped by various external influences. Cultural differences are a prime example: while highly spiced dishes are popular in many South Asian countries, milder flavors are often preferred in Western Europe. This project examines similar external factors that shape beer preferences, aiming to uncover what truly drives an individual’s taste in beers. We analyze how different beer characteristics are appreciated across selected countries and assess whether users from certain countries are more generous in their ratings compared to users from other countries. We also evaluate whether the origin of a beer biases the ratings it receives. Additionally, we investigate how seasonal variations impact the enjoyment of specific beer categories, and how user rating tendencies evolve as they gain experience. By identifying these “external” influences, we hope to help beer enthusiasts better understand their preferences and make choices based more on intrinsic qualities, ultimately improving their sensory experience and enjoyment of beer.

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

We use the average rating of certain beer styles as features of a given country and use a clustering algorithm to determine if certain countries cluster together in terms of beer style preferences. We also use the time information contained in beer reviews to determine whether regional beer style preferences remain stable over time—supporting the hypothesis that they are influenced by culture—or if they fluctuate, suggesting other contributing factors.

*Importance of specific beer attributes*

To evaluate the overall importance of various beer attributes in determining the final rating, we perform a linear regression analysis using attribute ratings as predictors and the final rating as the outcome variable, considering data from all countries combined. To account for potential confounders, the model also includes additional independent variables, specifically the average rating for the corresponding beer style and the average rating for the corresponding brewery. By comparing the coefficients of each attribute, we assess their relative influence on the final rating. To explore how the importance of these attributes varies across countries, we repeat the analysis for individual countries and examine the distribution of attribute coefficients across them.

**Location-related biases in ratings**

*Cultural biases*

To see whether users from certain countries are more generous or more critical in their ratings compared to users from other countries, we use a logistic regression model where the dependent variable is the reviewer’s country, and the independent variables are potentially relevant confounders, namely average rating for the beer style, brewery average rating, and the number of reviews given by the user. The resulting propensity scores represent the likelihood of a review being associated with a particular country given these confounders. Using these propensity scores, we match individual reviews from reviewers in one country with reviews from reviewers in another country that have similar propensity scores. This matching ensures that the paired reviews are comparable in terms of confounders, so that any differences in ratings are attributable to the reviewer’s country rather than other factors. We then compare paired rating differences between all pairs of user location. To achieve this, we group the matched reviews by country pairs and perform a paired t-test for each group to test whether the mean difference in ratings is significantly different from zero. Only country pairs with at least 10 matched reviews are included in the analysis.

Our analysis suggests that, after accounting for confounders, users from certain countries rate more generously than others. Clear trends in rating differences for certain country pairs suggest a statistically and practically significant 'cultural bias' among beer reviewers, which may meaningfully impact beer reviews.

*Beer origin bias*

To assess whether users rate domestic beers higher than foreign ones, we first identify users who have rated both domestic and foreign beers and filter for the reviews from those users. We label each review as either domestic (reviewing a beer produced in the reviewer’s country) or foreign. For each user, we match their reviews of domestic and foreign beers to ensure that comparisons are made between beers reviewed by the same user. We also make sure that matched reviews review beers of the same style to account for the effect of beer style on ratings. We then perform a paired t-test to determine whether the mean difference in ratings within pairs is significantly different from zero. We repeat this analysis focusing only on reviews from beer enthusiasts —users who have written a substantial number of reviews— who might prioritize intrinsic qualities of the beers over external factors such as location.

**Other biases**

*Seasonal biases*

To examine how seasonal changes influence the ratings of different beer styles, we use the time information contained in ratings to identify the season during which each rating was posted, taking into account the location of the user (Northern hemisphere, Southern hemisphere or equatorial area) to accurately determine the season. For simplicity, we only perform the analysis on users from the 10 countries with the highest number of reviews. 

We then perform a linear regression analysis with the final rating as the dependent variable and key predictors, namely season, average rating for the beer style, average rating for the brewery, user’s average rating and ABV as the independent variables. We also include attribute ratings as independent variables, as previous parts of the analysis revealed that they are important predictors of beer ratings. Since, as explained thereafter, we use the model to make predictions, we need the model to have high R2 and low RMSE for the predictions to be accurate and interpretable. This is why we try to include all key predictors in the model. The regression coefficients for the season variable reveal whether ratings are significantly higher or lower in certain seasons after accounting for confounders. Using these coefficients, we calculate predicted ratings for each beer style in each season and visualize the results with line charts, showing seasonal trends for different beer styles. This allows us to visualize how ratings fluctuate with the seasons and how these patterns differ by beer style.

Our analysis shows that while seasons appear to have a statistically significant impact on beer ratings, the effect is not significant from a practical standpoint. Most beer styles receive consistent ratings across all seasons.


*Experience bias*

To analyze how users’ rating tendencies evolve with experience, we focus on users who have posted a substantial number of reviews, based on a chosen threshold. For each user, we sort their reviews chronologically and assign an "experience level" to each rating based on the number of reviews they had posted up to that point. These levels are predefined and consistent across all users: new reviewer (first n reviews), amateur (from the n+1th to the oth review), and expert (from the o+1th review onward).To compare rating tendencies across all users, we fit a linear regression model to quantify the effect of experience level on ratings while adjusting for potential confounders. The dependent variable is the final rating, and the independent variables include experience level, the average rating for the beer style, the average rating for the brewery, and the average rating given by the user. The coefficients for the experience level provide insight into how ratings evolve with experience after accounting for confounders. To examine how ratings change within individual users, we perform a paired analysis. We match reviews labeled as “new reviewer” and “expert” within each user, ensuring the paired reviews correspond to the same beer style and brewery to control for confounders. We then use a paired t-test to determine whether the mean difference in ratings within these matched pairs is significant. This combination of regression modeling and paired analysis allows us to assess both overall trends and individual-level changes in rating behavior with experience.

Both the linear regression and paired analysis show that user experience level has a statistically significant but numerically small effect on ratings, with more experienced users being slightly more critical. The small impact is reflected in the low regression coefficients and minimal mean differences in the paired analysis. Thus, while user experience level has a detectable influence on ratings, it is not the dominant factor driving rating behavior. Other factors, such as beer style, brewery reputation, and user generosity, play a much larger role.


### Proposed timeline:
**Week 7:** project choice

**Week 8:** data loading and cleaning

**Week 9:** verification of project feasibility (ensuring there is enough data for the different tasks), writing of the README, and beginning implementation of some analysis components if time permits 

**Weeks 12 and 13:** implementation of the analyses described in "Methods"

**Week 14:** writing of the data story


### Organization within the team:
**Data loading:** Athénaïs

**Data cleaning:** Athénaïs

**Beer style preferences:** Orkun

**Importance of specific beer attributes:** Samet

**Cultural biases:** Athénaïs

**Beer origin bias:** Melis

**Seasonal biases:** Begüm

**Experience bias:** Athénaïs

**Data story:** everyone (everyone will add complete the data story with elements related to the part they worked on)


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
│   │   ├── data_utils.py               <- Helper functions used in results.ipynb
│   │   └── modules_utils.py            <- Imports required for results.ipynb
│   ├── scripts                 <- Scripts directory
│   │   └── review_parser.py            <- Script that processes each ratings.txt file by dividing it into parts, parsing each part, and
│                                          saving as JSON
│
├── tests                       <- Tests of any kind
│
│
├── results.ipynb               <- Notebook containing our analyses (calls helper functions from data_utils.py
│                                  and imports utilities from modules_utils.py)
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
