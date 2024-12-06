import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import pearsonr
import statsmodels.api as sm  


def calculate_trend(df): 
    X = df['year'].values.reshape(-1, 1)
    y = df['avg_rating'].values
    if len(X) > 1:  # Need more than one year to calculate trend
        model = LinearRegression()
        model.fit(X, y)
        trend = model.coef_[0] 
    else:
        trend = np.nan  # Not enough data to calculate trend
    return trend


def merge_data_(data):

    style_country_counts = {}

    unique_beer_styles = data['style'].unique()
    
    for style in unique_beer_styles:
    
        style_data = data[data['style'] == style]     # Filter data for the current beer style
        
        country_counts = style_data['user_location'].value_counts()       # Count the number of ratings by country for this style
        
        style_country_counts[style] = country_counts      # Store the result in the dictionary with the beer style as the key
    
    merged_df = pd.DataFrame(style_country_counts).transpose().fillna(0) # Convert the dictionary to a DataFrame

    return merged_df


def plot_actual_vs_predicted_(y_actual, y_pred, title='Actual vs Predicted', cmap='Reds', line_color='red', line_style='--', alpha=0.8, point_size=30, fig_size=(6, 4)):
    """
    Plots a compraison of actual values and predicted values  from the chosen training model (linear, lasso or ridge).

    Parameters:
    y_actual (array): array of actual values of the target variable.
    y_pred (array): Array of predicted target values from the model.
    title (str): Title of the plot. Defaults to 'Actual vs Predicted'.
    cmap (str): Colormap to use for the scatter plot. Defaults to 'Greys'.
    line_color (str): Color of the line of equality. Defaults to 'red'.
    line_style (str): Line style of the line of equality. Defaults to '--'.
    alpha (float): Transparency level of the scatter points. Defaults to 0.8.
    point_size (int): Size of the scatter points. Defaults to 30.
    fig_size (tuple): Size of the figure in inches (width, height). Defaults to (6, 4).

    """
 
    fig, ax = plt.subplots(figsize=fig_size)
    scatter = ax.scatter(y_actual, y_pred, c=y_actual, cmap=cmap, alpha=alpha, s=point_size, edgecolor='k')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Actual Values")

    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color=line_color, linestyle=line_style,  linewidth=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.show()


def plot_residuals(y_actual, y_pred, title='Residual Plot', cmap='coolwarm', line_color='red', alpha=0.8, point_size=30, fig_size=(6, 4), show_hist=False):
    """
    Plots the residuals (difference between actual values and predicted values) with custom styling, using a colormap.

    Parameters:
    y_actual (array): Array of actual values.
    y_pred (array): Array of predicted values.
    title (str): Title of the plot.
    cmap (str): Colormap to use for the scatter plot. Default cmap is 'coolwarm'.
    line_color (str): Color of the zero line on the residual plot. Default color'red'.
    alpha (float): Transparency level for scatter plot points. Default value is 0.8.
    point_size (int): Size of scatter plot points. Default value is 30 .
    fig_size (tuple): Size of the figure in inches (width, height). Default value is False.
    show_hist (bool): If True, overlays a histogram of residuals on the plot. Default value is True.

    """
    residuals= y_actual- y_pred
    fig, ax = plt.subplots(figsize=fig_size)
    #Creating a scatter plot of actual values versus residual values so that color changes according to the magnitude of the residaul
    scatter = ax.scatter(y_actual, residuals, c=residuals, cmap=cmap, alpha=alpha, s=point_size, edgecolor='k')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Residuals")
    
    # Drawing the line for zero value and namig the axes
    ax.axhline(y=0, color=line_color, linestyle='--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    # Optional histogram overlay
    if show_hist:
        ax_hist = ax.twinx()
        ax_hist.hist(residuals, bins=20, alpha=0.2, color='grey', orientation='horizontal')
        ax_hist.set_yticks([])

    plt.show()


def advanced_linear_regression(X, y, model_type='linear', make_plots=True, alphas=np.logspace(-4, 4, 100), 
                               scale_data=True, test_size=0.3, random_state=66, cross_validate=False, 
                               cv_folds=5, return_model=False, print_summary=True):
    """
    Trains and evaluates a regression model with options for scaling, cross-validation, and different model types.
    Optionally prints model summary and returns the trained model.

    Parameters:
    X (pandas dataframe or numpy array): Matrix of independent variables used to predict the target variable.
    y (pandas dataframe or numpy array): Target variable vector.
    model_type (str): Type of regression model to use. Options are 'linear', 'lasso', and 'ridge'. Default is 'linear'.
    alphas (array): Array of alpha values for regularization, used for Lasso and Ridge regression.
    scale_data (bool): Option to standardize features before training. Default is True.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.3.
    random_state (int): Seed for random number generator to ensure reproducibility. Default is 66.
    cross_validate (bool): Whether to perform cross-validation. Defaults to False.
    cv_folds (int): Number of cross-validation folds if cross_validate is True. Default is 5.
    return_model (bool): Whether to return the trained model. Defaults to False.
    print_summary (bool): Whether to print the model summary. Defaults to True.

    Returns:
    dict: A dictionary containing training and testing metrics:
        - 'train_mse': Mean squared error on the training set.
        - 'train_r2': R-squared score on the training set.
        - 'test_mse': Mean squared error on the test set.
        - 'test_r2': R-squared score on the test set.
        - 'train_pearson': Pearson correlation coefficient on the training set.
        - 'test_pearson': Pearson correlation coefficient on the test set.
    model (optional): The trained regression model if return_model=True.

    Raises:
    ValueError: If model_type is not one of 'linear', 'lasso', or 'ridge'.
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Optionally scale the data
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train the model
    if model_type == 'linear':
        # Add constant for statsmodels
        X_train_sm = sm.add_constant(X_train, has_constant='add')
        model = sm.OLS(y_train, X_train_sm).fit()
        if print_summary:
            print(model.summary())
        X_test_sm = sm.add_constant(X_test, has_constant='add')
        y_train_pred = model.predict(X_train_sm)
        y_test_pred = model.predict(X_test_sm)
    elif model_type == 'lasso':
        model = LassoCV(alphas=alphas, cv=cv_folds, random_state=random_state)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        if print_summary:
            print(f"Optimal alpha for Lasso: {model.alpha_}")
    elif model_type == 'ridge':
        model = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=cv_folds)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        if print_summary:
            print(f"Optimal alpha for Ridge: {model.alpha_}")
    else:
        raise ValueError("Invalid model_type. Choose 'linear', 'lasso', or 'ridge'.")

    # Calculate metrics
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

    # Perform cross-validation if specified
    if cross_validate:
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores
        print(f"Cross-Validation MSE: {cv_mse.mean()} ± {cv_mse.std()}")

    # Optionally make plots
    if make_plots:
        plot_residuals(y_train, y_train_pred, title=f'Residuals {model_type.capitalize()} Regression Train')
        plot_actual_vs_predicted_(y_train, y_train_pred, title=f'Actual vs Predicted {model_type.capitalize()} Regression Train')
        plot_residuals(y_test, y_test_pred, title=f'Residuals {model_type.capitalize()} Regression Test')
        plot_actual_vs_predicted_(y_test, y_test_pred, title=f'Actual vs Predicted {model_type.capitalize()} Regression Test')

    # Return metrics and optionally the model
    if return_model:
        return metrics, model
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


def plot_coefficients(coefficients, conf_int, title, bar_color, ci_color='black'):
    """
    Plots the coefficients of a regression model as bars along with their confidence intervals.

    Parameters:
    coefficients (pandas.Series): The regression coefficients. The index should correspond to variable names, and   values should be the estimated coefficients.
    conf_int (pandas.DataFrame): A DataFrame containing the lower and upper bounds of the confidence intervals for each coefficient. The first column should be the lower bound, and the second column should be the upper bound.
    title (str): The title of the plot.
    bar_color (str): The color of the bars representing the coefficient values.
    ci_color (str): The color of the lines and shading representing the confidence intervals.

    Returns:
    None. Displays the coefficient plot with bars for coefficients and confidence intervals as error bars.
    """
    # exclude the constant term if present
    if 'const' in coefficients.index:
        coefficients = coefficients.drop('const')
        conf_int = conf_int.drop('const')
    
    # extract confidence interval ranges
    lower_error = coefficients - conf_int[0]
    upper_error = conf_int[1] - coefficients
    
    # plot the coefficients as bars
    plt.figure(figsize=(8, 6))
    plt.bar(coefficients.index, coefficients.values, color=bar_color, alpha=0.8, label="Coefficient")
    
    # add confidence intervals as error bars
    plt.errorbar(coefficients.index, coefficients.values, 
                 yerr=[lower_error, upper_error], fmt='o', color=ci_color, label="95% CI")

    plt.axhline(0, color='gray', linestyle='--', linewidth=1) # horizontal line at y=0 for reference
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

