import pandas as pd

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
