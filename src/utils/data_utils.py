import pandas as pd
import re

def assign_experience_level(df, new_reviewer_threshold, amateur_threshold):
    """
    Assigns an experience level to each review based on the number of reviews posted by the user up to that review.

    Parameters:
    df (pd.DataFrame): DataFrame containing user reviews including columns 'user_id' and 'date'
    new_reviewer_threshold (int): number of reviews to be considered a new reviewer
    amateur_threshold (int): number of reviews to be considered an amateur before becoming an expert

    Returns:
    pd.DataFrame: updated DataFrame with an 'experience_level' column
    """

    df = df.sort_values(by=['user_id', 'date']).copy() # sort the dataframe by user and date

    df['experience_level'] = 'new_reviewer' # add a new column corresponding to the experience level

    # iterate over each user and assign experience levels to reviews based on the number of reviews up to the considered review
    for user_id, group in df.groupby('user_id'):
        review_nb = group.shape[0]  # total number of reviews for the considered user
        # assign 'new_reviewer' to the first new_reviewer_threshold reviews
        df.loc[group.index[:new_reviewer_threshold], 'experience_level'] = 'new_reviewer'
        
        # assign 'amateur' to reviews between new_reviewer_threshold and amateur_threshold
        df.loc[group.index[new_reviewer_threshold:amateur_threshold], 'experience_level'] = 'amateur'
        
        # assign 'expert' to the remaining reviews beyond the amateur_threshold
        df.loc[group.index[amateur_threshold:], 'experience_level'] = 'expert'
    
    return df



def clean_location_column(df):
    """
    Cleans the 'location' column by:
    - removing leading and trailing spaces
    - removing embedded HTML links
    - standardizing entries starting with 'United States' to 'United States'

    Parameters:
    df (pd.DataFrame): dataframe containing a 'location' column

    Returns:
    pd.DataFrame: updated dataframe with cleaned 'location' column
    """
    # make sure that a 'location' column is present
    if 'location' in df.columns:

        # remove leading and trailing spaces
        df['location'] = df['location'].astype(str).str.strip()

        # remove rows where 'location' contains '</a>' (indicating an embedded link)
        df = df[~df['location'].str.contains('</a>', na=False)]
        
        # standardize locations starting with 'United States' to 'United States'
        df['location'] = df['location'].apply(lambda x: 'United States' if x.startswith('United States') else x)
    
    return df