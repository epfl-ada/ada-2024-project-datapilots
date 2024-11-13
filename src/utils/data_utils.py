import pandas as pd
import re
import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import pearsonr
import statsmodels.api as sm  




def merge_data_(data):

    style_country_counts = {}

    unique_beer_styles = data['style'].unique()
    
    for style in unique_beer_styles:
    
        style_data = data[data['style'] == style]     # Filter data for the current beer style
        
        country_counts = style_data['user_location'].value_counts()       # Count the number of ratings by country for this style
        
        style_country_counts[style] = country_counts      # Store the result in the dictionary with the beer style as the key
    
    merged_df = pd.DataFrame(style_country_counts).transpose().fillna(0) # Convert the dictionary to a DataFrame

    return merged_df


def plot_actual_vs_predicted_(y_actual, y_pred, title='Actual vs Predicted'):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.3)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

def plot_residuals(y_actual, y_pred, title='Residual Plot'):
    residuals = y_actual - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()


def advanced_linear_regression(X, y, model_type='linear', alphas=np.logspace(-4, 4, 100), scale_data=True, test_size=0.3, random_state=42, cross_validate=False, cv_folds=5):
    """Train and evaluate a regression model with options for scaling, cross-validation, and different model types."""
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scaling if requested
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Model selection and training
    if model_type == 'linear':
        
        X_train_sm = sm.add_constant(X_train)  
        model = sm.OLS(y_train, X_train_sm).fit()
        
        
        print(model.summary())
        
        y_train_pred = model.predict(X_train_sm)
        y_test_pred = model.predict(sm.add_constant(X_test))
        
    elif model_type == 'lasso':
        model = LassoCV(alphas=alphas, cv=cv_folds, random_state=random_state)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        print(f"Optimal alpha for Lasso: {model.alpha_}")
        
    elif model_type == 'ridge':
        model = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=cv_folds)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        print(f"Optimal alpha for Ridge: {model.alpha_}")
        
    else:
        raise ValueError("Invalid model_type. Choose 'linear', 'lasso', or 'ridge'.")

   
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_pearson': pearsonr(y_train, y_train_pred)[0],
        'test_pearson': pearsonr(y_test, y_test_pred)[0]
    }
    print("Training MSE:", metrics['train_mse'], "Training R2:", metrics['train_r2'], "Training Pearson Correlation:", metrics['train_pearson'])
    print("Testing MSE:", metrics['test_mse'], "Testing R2:", metrics['test_r2'], "Testing Pearson Correlation:", metrics['test_pearson'])

    if cross_validate:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores
        print(f"Cross-Validation MSE: {cv_mse.mean()} ± {cv_mse.std()}")

    
    plot_residuals(y_train, y_train_pred, title=f'Residuals {model_type.capitalize()} Regression Train')
    plot_actual_vs_predicted_(y_train, y_train_pred, title=f'Actual vs Predicted {model_type.capitalize()} Regression Train')
    plot_residuals(y_test, y_test_pred, title=f'Residuals {model_type.capitalize()} Regression Test')
    plot_actual_vs_predicted_(y_test, y_test_pred, title=f'Actual vs Predicted {model_type.capitalize()} Regression Test')

    return metrics
    


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
    - standardizing entries starting with 'Canada' to 'Canada'
    - replacing locations with the structure 'United Kingdom, [second word]' by '[second word]'

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

        # standardize locations starting with 'Canada' to 'Canada'
        df['location'] = df['location'].apply(lambda x: 'Canada' if x.startswith('Canada') else x)

        # replace 'United Kingdom, [second word]' with '[second word]'
        df['location'] = df['location'].apply(lambda x: x.split(', ')[1] if x.startswith('United Kingdom,') else x)
    
    return df


def get_season(date):
    """
    Determines the season for a given date, assuming seasons based on the Northern Hemisphere.

    Parameters:
    date (datetime): a datetime object representing the date

    Returns:
    str: the season corresponding to the month of the date
         Returns 'Winter' for December, January, and February.
         Returns 'Spring' for March, April, and May.
         Returns 'Summer' for June, July, and August.
         Returns 'Fall' for September, October, and November.
 
    """
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

