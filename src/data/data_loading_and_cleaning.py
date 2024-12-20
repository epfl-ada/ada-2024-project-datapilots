import os

os.chdir("src/utils")
exec(open("modules_utils.py").read())

os.chdir("../../")

#DATASET

"""

The dataset used in this analysis consists of beer reviews from two beer rating websites,**BeerAdvocate** and **RateBeer**, for a period ranging from 2001 to 2017. For each website, we have 5 files:
- users.csv: metadata about reviewers
- beers.csv : metadata about reviewed beers
- breweries.csv : metadata about breweries
- ratings.txt : all reviews given by users, including numerical ratings and sometimes textual reviews
- reviews.txt : only reviews given by users that include both numerical ratings and textual reviews

In our analysis, we will not use textual reviews. Thus, we will only use ratings.txt files and not reviews.txt files, as we will use all reviews, whether or not they include textual reviews.

"""
###################################################################################################################################################################################################################
###################################################################################################################################################################################################################

# LOAD DATA INTO DATAFRAMES
###########################

"""
The .csv files are not too large and can efficiently be loaded into DataFrames.

"""

BA_DATA_FOLDER = 'data/BeerAdvocate/'
RB_DATA_FOLDER = 'data/RateBeer/'

BA_USERS = BA_DATA_FOLDER+"users.csv"
BA_BEERS = BA_DATA_FOLDER+"beers.csv"
BA_BREWERIES = BA_DATA_FOLDER+"breweries.csv"

RB_USERS = RB_DATA_FOLDER+"users.csv"
RB_BEERS = RB_DATA_FOLDER+"beers.csv"
RB_BREWERIES = RB_DATA_FOLDER+"breweries.csv"

ba_users = pd.read_csv(BA_USERS)
ba_beers = pd.read_csv(BA_BEERS)
ba_breweries = pd.read_csv(BA_BREWERIES)

rb_users = pd.read_csv(RB_USERS)
rb_beers = pd.read_csv(RB_BEERS)
rb_breweries = pd.read_csv(RB_BREWERIES)

"""
On the other hand, the ratings.txt files are extremely large, and trying to load them directly into DataFrames leads to kernel freezes. In order to circumvent this problem, we wrote a script (review_parser.py, located in src/scripts), which processes each rating file by dividing it into parts, parsing each part, and saving as JSON. In the notebook, we then load the different JSON files into DataFrames, that we concatenate. Dividing the large .txt files into smaller JSON chunks and then loading each chunk separately, avoids trying to load the entire file into memory at once, which can cause kernel freezes due to memory overload. In addition, JSON is a format that pandas can read efficiently.

"""

# Load BeerAdvocate ratings stored in json files into a single DataFrame
ba_json_files = glob.glob(BA_DATA_FOLDER+'*.json')
ba_df_list = [pd.read_json(file) for file in ba_json_files]
ba_ratings = pd.concat(ba_df_list, ignore_index=True)

# Load RateBeer ratings stored in json files into a single DataFrame
rb_json_files = glob.glob(RB_DATA_FOLDER+'*.json')
rb_df_list = [pd.read_json(file) for file in rb_json_files]
rb_ratings = pd.concat(rb_df_list, ignore_index=True)


###################################################################################################################################################################################################################
###################################################################################################################################################################################################################


# BASIC CLEANING
################

"""
Let us start by removing columns in the different Dataframes that we will not use in our analysis.

The following rows will not be used in our analysis:
nbr_reviews, ba_score, bros_score, abv, avg_computed, zscore, nbr_matched_valid_ratings and avg_matched_valid_ratings, overall_score and style_score.

Let us remove them.

"""

useless_columns_ba = ['nbr_reviews', 'ba_score', 'bros_score', 'abv', 'avg_computed', 'zscore', 'nbr_matched_valid_ratings', 'avg_matched_valid_ratings']
ba_beers = ba_beers.drop(columns=useless_columns_ba)

useless_columns_rb = [col for col in useless_columns_ba if col not in ['nbr_reviews','ba_score', 'bros_score']] + ['overall_score', 'style_score']
rb_beers = rb_beers.drop(columns=useless_columns_rb)

"""
We will also not use the timestamps indicating the time when users joined the platforms, so let us remove this as well.

"""

ba_users = ba_users.drop(columns='joined')
rb_users = rb_users.drop(columns='joined')

"""

## Verifying value types

Let us verify that the values in the different columns of the different Dataframes have the appropriate type.

print(ba_beers.dtypes,'\n','\n',rb_beers.dtypes)
print(ba_users.dtypes,'\n','\n',rb_users.dtypes)
print(ba_breweries.dtypes,'\n','\n',rb_breweries.dtypes)
print(ba_ratings.dtypes,'\n','\n',rb_ratings.dtypes)

##Conclusion : The types of the values in the different columns of the different Dataframes seem appropriate.

"""

# DEALING WITH MISSING VALUES

#############################

"""
Let us verify that all the beers in the beer DataFrames have at least received 1 review.

"""

# Get the number of beers with 0 reviews
# Beer Advocate
ba_beers_without_reviews = ba_beers[ba_beers['nbr_ratings'] == 0].shape[0]
#print('Number of beers with no reviews (Beer Advocate):',ba_beers_without_reviews)

# Rate Beer
rb_beers_without_reviews = rb_beers[rb_beers['nbr_ratings'] == 0].shape[0]
#print('Number of beers with no reviews (Rate Beer):',rb_beers_without_reviews)

# Result 
"""
Number of beers with no reviews (Beer Advocate): 32841
Number of beers with no reviews (Rate Beer): 45391

We can see that in Both Dataframes, there are many beers with no reviews. These beers are useless for our analysis and we can remove them from the DataFrames.

"""

# Remove beers with zero reviews
# Beer Advocate
ba_beers = ba_beers[ba_beers['nbr_ratings'] != 0]
# Rate Beer
rb_beers = rb_beers[rb_beers['nbr_ratings'] != 0]

"""
Dealing with reviews with no final rating. 

Let us now determine if some reviews lack a final rating. If that is the case, we will remove them from the rating DataFrames as we will need final ratings in our analysis. The final rating, unlike the overall rating, cannot be calculated using aspect ratings, so we cannot retrieve missing final rating values. Some reviews may lack certain aspect ratings, but we will not remove them as if they do have a final rating, we will be able to use them in many parts of our analysis.

"""

# get the number of reviews with Nan in the 'rating' column
# Beer Advocate
#print('Number of reviews lacking a final rating (Beer Advocate):',ba_ratings['rating'].isna().sum())
# Rate Beer
#print('Number of reviews lacking a final rating (Rate Beer):',rb_ratings['rating'].isna().sum())

# Result
"""
Number of reviews lacking a final rating (Beer Advocate): 2
Number of reviews lacking a final rating (Rate Beer): 2

There are only 2 reviews lacking a final rating for each website, but let us remove them anyway.

"""

# remove reviews with no final rating
ba_ratings = ba_ratings.dropna(subset=['rating'])
rb_ratings = rb_ratings.dropna(subset=['rating'])

# CLEANING LOCATION INFORMATION
###############################

"""
Let us have a closer look at the values present in the 'location' column of user and brewery DataFrames.

"""

"""
 ba_users['location'].value_counts()
 
 location
United States, California       11638
United States, Pennsylvania      8689
United States, New York          7432
United States, Illinois          6866
United States, Massachusetts     6658
                                ...  
Angola                              1
Kazakhstan                          1
Tokelau                             1
Sri Lanka                           1
Sint Maarten                        1
Name: count, Length: 194, dtype: int64

 rb_users['location'].value_counts()

location
Canada                         3255
United States, California      2804
England                        2734
Poland                         1701
United States, Pennsylvania    1632
                               ... 
Liechtenstein                     1
Lesotho                           1
East Timor                        1
Falkland Islands                  1
Tibet                             1
Name: count, Length: 222, dtype: int64

 ba_breweries['location'].value_counts()
 
location
Germany                                                                                                                                                                          1431
England                                                                                                                                                                           997
United States, California                                                                                                                                                         929
Canada                                                                                                                                                                            775
United States, Washington                                                                                                                                                         411
                                                                                                                                                                                 ... 
Andorra                                                                                                                                                                             1
Botswana                                                                                                                                                                            1
Turks and Caicos Islands                                                                                                                                                            1
Papua New Guinea                                                                                                                                                                    1
United States</a> | <a href="http://maps.google.com/maps?oi=map&q=%2C+US" target="_blank">map</a><br><a href="http://fullsailbrewing.com" target="_blank">fullsailbrewing.com       1
Name: count, Length: 297, dtype: int64

 rb_breweries['location'].value_counts()

location
England                      2124
Germany                      1999
Italy                        1051
Canada                        884
United States, California     867
                             ... 
Senegal                         1
Mozambique                      1
French Polynesia                1
Papua New Guinea                1
Gabon                           1
Name: count, Length: 267, dtype: int64


# see all the possibilities for locations containing 'Wales', 'Scotland', 'England' or 'Ireland'
ba_breweries[ba_breweries['location'].str.contains('Wales|Scotland|England|Ireland', case=False, na=False)]['location'].value_counts()

location
England                     997
Scotland                    104
Ireland                      84
Wales                        58
United Kingdom, England      32
Northern Ireland             25
United Kingdom, Scotland      3
United Kingdom, Wales         2
Name: count, dtype: int64

 # see all the possibilities for locations starting with 'Canada'
ba_breweries[ba_breweries['location'].str.startswith('Canada', na=False)]['location'].value_counts()

location
Canada                               775
Canada, Quebec                        28
Canada, Ontario                       25
Canada, British Columbia              12
Canada, Nova Scotia                    6
Canada, Alberta                        6
Canada, Manitoba                       3
Canada, Saskatchewan                   2
Canada, New Brunswick                  2
Canada, Newfoundland and Labrador      1
Name: count, dtype: int64


"""


""" 
Conclusion:

We can make 4 important observations: 
- US locations all contain state information
- some locations contain embedded HTML links
- Canada locations contain province information
- UK countries are represented in 2 different ways: either 'United Kingdom, *country*' or '*country*'

We will remove both these HTML links and the state and province information, as we will not use it in our analysis. We will also make sure that UK countries are represented in only 1 way. We will do so by applying a function named 'clean_location_column' which cleans the location information and that we have written in src/utils/data_utils.py().

"""

# LOCATION CLEANING

# Clean location information in user and brewery dataframes
# Beer Advocate
ba_users = clean_location_column(ba_users)
ba_breweries = clean_location_column(ba_breweries)

# Rate Beer
rb_users = clean_location_column(rb_users)
rb_breweries = clean_location_column(rb_breweries)


# ADDITION OF USEFUL COLUMNS

"""
To streamline our analysis and avoid redundant computations, we will calculate certain statistics that will be used in several parts of the analysis once and store them as new columns in the respective dataframes.

Let us first add a column to the rating dataframes corresponding to the average rating given by the user who wrote the review.

"""

# calculate the average rating for each user
average_ratings_by_user_ba = ba_ratings.groupby('user_id')['rating'].mean().reset_index()
average_ratings_by_user_ba.rename(columns={'rating': 'user_avg_rating'}, inplace=True)

average_ratings_by_user_rb = rb_ratings.groupby('user_id')['rating'].mean().reset_index()
average_ratings_by_user_rb.rename(columns={'rating': 'user_avg_rating'}, inplace=True)

# add the average ratings by user to the rating dataframes
ba_ratings = ba_ratings.merge(average_ratings_by_user_ba, on='user_id', how='left')
rb_ratings = rb_ratings.merge(average_ratings_by_user_rb, on='user_id', how='left')

#ba_ratings.head()

"""
Let us now add a column to the rating dataframes corresponding to the average rating of all beers coming from the brewery that produced the reviewed beer. This metric can be used as a proxy for brewery reputation and will be used in certain parts of the analysis to account for the fact that some breweries might have established reputations that bias user ratings.

"""

# calculate the average rating for each brewery
average_ratings_by_brewery_ba = ba_ratings.groupby('brewery_id')['rating'].mean().reset_index()
average_ratings_by_brewery_ba.rename(columns={'rating': 'brewery_avg_rating'}, inplace=True)

average_ratings_by_brewery_rb = rb_ratings.groupby('brewery_id')['rating'].mean().reset_index()
average_ratings_by_brewery_rb.rename(columns={'rating': 'brewery_avg_rating'}, inplace=True)

# add the average ratings by brewery to the rating dataframes
ba_ratings = ba_ratings.merge(average_ratings_by_brewery_ba, on='brewery_id', how='left')
rb_ratings = rb_ratings.merge(average_ratings_by_brewery_rb, on='brewery_id', how='left')

#ba_ratings.head()

"""
We will now add a column to the rating dataframes corresponding to the average rating of all beers with the same style as the reviewed beer. 

"""

# calculate the average rating for each beer style
average_ratings_by_style_ba = ba_ratings.groupby('style')['rating'].mean().reset_index()
average_ratings_by_style_ba.rename(columns={'rating': 'style_avg_rating'}, inplace=True)

average_ratings_by_style_rb = rb_ratings.groupby('style')['rating'].mean().reset_index()
average_ratings_by_style_rb.rename(columns={'rating': 'style_avg_rating'}, inplace=True)

# add the average ratings by style to the rating dataframes
ba_ratings = ba_ratings.merge(average_ratings_by_style_ba, on='style', how='left')
rb_ratings = rb_ratings.merge(average_ratings_by_style_rb, on='style', how='left')

#ba_ratings.head()

"""
Finally, we will add a column to the rating dataframes corresponding to the number of reviews given by the user who wrote the review.

"""

# merge the rating dataframes with the users dataframes on user_id
ba_ratings = ba_ratings.merge(ba_users[['user_id', 'nbr_reviews']], on='user_id', how='left')
rb_ratings = rb_ratings.merge(rb_users[['user_id', 'nbr_ratings']], on='user_id', how='left')

ba_ratings = ba_ratings.rename(columns={'nbr_reviews': 'user_nb_reviews'})
rb_ratings = rb_ratings.rename(columns={'nbr_ratings': 'user_nb_reviews'})

#ba_ratings.head()

# LOCATION SPECIFIC CLEANING

"""
Several parts of our analysis will involve comparing data from different countries and will require working with the location information in the user DataFrames. In these parts, we will only consider countries with a 'sufficiently large' number of reviewers, as only a few reviewers may not be representative of an entire country. We will thus filter out countries that we consider to have not enough reviewers. We decide arbitrarily to filter out countries with less than 50 different reviewers.

In this part of the data cleaning, we will work on a copy of the original DataFrames, and we will use these copies only for the parts of the analysis where we compare certain countries. Indeed, this filtering is not relevant for the parts that do not involve comparing certain countries.

"""

# create copies of the user and rating DataFrames, which will undergo filtering related to the location information
ba_users_loc_filtered = ba_users.copy()
rb_users_loc_filtered = rb_users.copy()
ba_ratings_loc_filtered = ba_ratings.copy()
rb_ratings_loc_filtered = rb_ratings.copy()

# Removing users with missing location information

"""
First, let us determine if some users are missing the location information.

# get the number of users with 'nan' in the 'location' column
# Beer Advocate
print('Number of users lacking the location information (Beer Advocate):',ba_users_loc_filtered[ba_users_loc_filtered['location'] == 'nan'].shape[0])
# Rate Beer
print('Number of users lacking the location information (Rate Beer):',rb_users_loc_filtered[rb_users_loc_filtered['location'] == 'nan'].shape[0])

Number of users lacking the location information (Beer Advocate): 31279
Number of users lacking the location information (Rate Beer): 19582

We can see that there are indeed some users missing the location information. Let us remove these users.

"""

ba_users_loc_filtered = ba_users_loc_filtered[ba_users_loc_filtered['location'] != 'nan']
rb_users_loc_filtered = rb_users_loc_filtered[rb_users_loc_filtered['location'] != 'nan']

# Filtering out users and reviews written by users from countries with too few users

"""
Let us first identify the countries with a number of users that is smaller than the threshold we set earlier, that is, 10 users.

"""

USER_NB_THRESHOLD = 50

# concatenate the 'location' columns from the 2 user dataframes
combined_locations = pd.concat([ba_users_loc_filtered['location'], rb_users_loc_filtered['location']])

# get the counts of each unique value in 'location'
location_counts = combined_locations.value_counts()

# keep only locations with less than 50 counts
locations_with_few_counts = location_counts[location_counts < USER_NB_THRESHOLD]

# Get the list of location values with less than 50 counts
countries_with_few_users = locations_with_few_counts.index.tolist()

"""
print("Countries with less than 50 users across both DataFrames:")
print(countries_with_few_users)

Countries with less than 50 users across both DataFrames:
['Aotearoa', 'Panama', 'Colombia', 'Bosnia and Herzegovina', 'Costa Rica', 'Cyprus', 'Luxembourg', 'Belarus', 'Dominican Republic', 'El Salvador', 'Peru', 'Moldova', 'Guatemala', 'Ecuador', 'Macedonia', 'Venezuela', 'Paraguay', 'Uruguay', 'Lebanon', 'Malaysia', 'Vietnam', 'Indonesia', 'Virgin Islands (U.S.)', 'Bahamas', 'Faroe Islands', 'Afghanistan', 'Malta', 'Andorra', 'Cambodia', 'Guam', 'Kenya', 'United Arab Emirates', 'Isle of Man', 'Antarctica', 'Cayman Islands', 'Bermuda', 'Bolivia', 'Honduras', 'Montenegro', 'Egypt', 'Tanzania', 'Uganda', 'Botswana', 'Zimbabwe', 'Albania', 'Tajikistan', 'Azerbaijan', 'Trinidad and Tobago', 'American Samoa', 'Tunisia', 'Jersey', 'Vanuatu', 'Pakistan', 'Jordan', 'Mauritius', 'Belize', 'Nicaragua', 'Barbados', 'Mozambique', 'Aruba', 'Uzbekistan', 'Nigeria', 'Palestine', 'Namibia', 'Abkhazia', 'Papua New Guinea', 'Armenia', 'Morocco', 'Nepal', 'Burkina Faso', 'Marshall Islands', 'Ascension Island', 'Vatican City', 'Hawaii', 'Algeria', 'Kosovo', 'Laos', 'Monaco', 'Bangladesh', 'Sint Maarten', 'Norfolk Island', 'Ghana', 'Slovakia', 'Saint Helena', 'Dem Rep of Congo', 'San Marino', 'Zambia', 'Iraq', 'Syria', 'Northern Mariana Islands', 'Kazakhstan', 'New Caledonia', 'Fiji Islands', 'Bouvet Island', 'Lesotho', 'Guernsey', 'Svalbard and Jan Mayen Islands', 'Bhutan', 'Ethiopia', 'Haiti', 'Antigua and Barbuda', 'Kyrgyzstan', 'Iran', 'Tokelau', 'Sudan', 'Anguilla', 'Oman', 'Liechtenstein', 'Yemen', 'East Timor', 'Falkland Islands', 'Virgin Islands (British)', 'Nagorno-Karabakh', 'Micronesia', 'Saint Lucia', 'Saint Vincent and The Grenadines', 'Tuvalu', 'Jamaica', 'Macau', 'British Indian Ocean Territory', 'Gibraltar', 'Montserrat', 'Saudi Arabia', 'Senegal', 'North Korea', 'Greenland', 'Mauritania', 'Christmas Island', 'Seychelles', 'Turkmenistan', 'Liberia', 'Togo', 'Solomon Islands', 'Angola', 'Mongolia', 'South Georgia and South Sandwich Islands', 'Equatorial Guinea', 'Burundi', 'Heard and McDonald Islands', 'Ivory Coast', 'Rwanda', 'Cuba', 'Sri Lanka', 'French Guiana', 'Malvinas', 'Tibet']

Filtering out those countries would leave us with 55 countries, which is enough to perform our analysis.

Ultimately, we would like to remove reviews from users coming from countries with too few reviewers. 
To do so, we will start by adding the user location information to review DataFrames. This will facilitate the filtering and analysis.


"""

# merge rating dataframes with user dataframes on 'user_id' to add the'location' column to rating dataframes as 'user_location'

# Beer Advocate
ba_ratings_loc_filtered = ba_ratings_loc_filtered.merge(ba_users_loc_filtered[['user_id', 'location']], on='user_id', how='left')
ba_ratings_loc_filtered = ba_ratings_loc_filtered.rename(columns={'location': 'user_location'})

# Rate Beer
rb_ratings_loc_filtered = rb_ratings_loc_filtered.merge(rb_users_loc_filtered[['user_id', 'location']], on='user_id', how='left')
rb_ratings_loc_filtered = rb_ratings_loc_filtered.rename(columns={'location': 'user_location'})

#ba_ratings_loc_filtered.head()
"""
We now must remove the reviews from users whose location is NaN. Such users correspond to the users lacking a ocation information that have been previously removed from the user dataframes, but which are still present in the review dataframes.

"""

# remove reviews from users for which the location is NaN
ba_ratings_loc_filtered = ba_ratings_loc_filtered.dropna(subset=['user_location'])
rb_ratings_loc_filtered = rb_ratings_loc_filtered.dropna(subset=['user_location'])

# remove reviews where 'user_location' is in the 'countries_with_few_users' list
ba_ratings_loc_filtered = ba_ratings_loc_filtered[~ba_ratings_loc_filtered['user_location'].isin(countries_with_few_users)]
rb_ratings_loc_filtered = rb_ratings_loc_filtered[~rb_ratings_loc_filtered['user_location'].isin(countries_with_few_users)]

# remove users for which 'location' is in the 'countries_with_few_users' list
ba_users_loc_filtered = ba_users_loc_filtered[~ba_users_loc_filtered['location'].isin(countries_with_few_users)]
rb_users_loc_filtered = rb_users_loc_filtered[~rb_users_loc_filtered['location'].isin(countries_with_few_users)]

