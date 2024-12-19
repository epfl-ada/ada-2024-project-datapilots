import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import statsmodels.api as sm 
from wordcloud import WordCloud
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from bisect import bisect_left

def generate_wordcloud_from_series(series,title, mask_image_path=None, ax=None):

    """
    Function for creating Wordcloud maps that uses pandas series dataframe
    
    """

    data_dict = series.to_dict() # Pandas series dataframe to dictionary
    
    mask = None
    if mask_image_path:
        mask = np.array(Image.open(mask_image_path))

    # Generate the workcloud 
    wordcloud = WordCloud(
        background_color="white",
        mask=mask,
        contour_width=1,
        contour_color="black",
        colormap="viridis",
        prefer_horizontal=0.9
    ).generate_from_frequencies(data_dict)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)

def region_mapping(merged_df):   # Function for adding region information to our data  
    
    """
    Sources: 
              - https://worldinmaps.com/geography-and-geology/world-regions/
              - https://vividmaps.com/world-map-region-definitions/
    
          Note: But eventually we have to change some countries regions (otherwise it some regions have only 1 countries)
    
    """

    country_region_mapping = {
        # North America
        'United States': 'North America',
        'Canada': 'North America',
    
        # Western Europe
        'England': 'Western Europe',
        'Netherlands': 'Western Europe',
        'Sweden': 'Western Europe',
        'Germany': 'Western Europe',
        'Belgium': 'Western Europe',
        'Denmark': 'Western Europe',
        'France': 'Western Europe',
        'Scotland': 'Western Europe',
        'Ireland': 'Western Europe',
        'Wales': 'Western Europe',
        'Northern Ireland': 'Western Europe',
        'Switzerland': 'Western Europe',
        'Austria': 'Western Europe',
        'Iceland': 'Western Europe',
        'Finland': 'Western Europe',
        'Norway': 'Western Europe',
    
        # Southern Europe
        'Italy': 'Southern Europe',
        'Spain': 'Southern Europe',
        'Portugal': 'Southern Europe',
        'Greece': 'Southern Europe',
        'Croatia': 'Southern Europe',
        'Slovenia': 'Southern Europe',
    
        # Eastern Europe
        'Poland': 'Eastern Europe',
        'Russia': 'Eastern Europe',
        'Hungary': 'Eastern Europe',
        'Czech Republic': 'Eastern Europe',
        'Romania': 'Eastern Europe',
        'Estonia': 'Eastern Europe',
        'Turkey': 'Eastern Europe',
        'Serbia': 'Eastern Europe',
        'Slovak Republic': 'Eastern Europe',
        'Ukraine': 'Eastern Europe',
        'Latvia': 'Eastern Europe',
        'Lithuania': 'Eastern Europe',
        'Bulgaria': 'Eastern Europe',
        'Israel': 'Eastern Europe',
        'South Africa': 'Eastern Europe',
        
        # Latin America
        'Brazil': 'Latin America',
        'Chile': 'Latin America',
        'Argentina': 'Latin America',
        'Mexico': 'Latin America',
        'Puerto Rico': 'Latin America',
    
        # Eastern Asia
        'Japan': 'Eastern Asia',
        'China': 'Eastern Asia',
        'South Korea': 'Eastern Asia',
        'Taiwan': 'Eastern Asia',
        'Hong Kong': 'Eastern Asia',
        'Philippines': 'Eastern Asia',
        'Thailand': 'Eastern Asia',
        'India': 'Eastern Asia',
        'Singapore': 'Eastern Asia',
    
        # Australia and Oceania
        'Australia': 'Australia and Oceania',
        'New Zealand': 'Australia and Oceania',
    }
    
    countries = merged_df.columns.tolist()  # Transforming column names to a list

    # Extracting countries and map them to regions
    countries = merged_df.columns.tolist()  # Transforming column names to a list
    country_region_df = pd.DataFrame({'country': countries})
    country_region_df['region'] = country_region_df['country'].map(country_region_mapping)
    
    # Transposing the DataFrame and add region information
    transposed_df = merged_df.T  # Now countries are rows, beer styles are columns
    transposed_df['region'] = transposed_df.index.map(country_region_mapping)  # Add the region information
    merged_with_regions = transposed_df.T
    
    # Reordering countries by their regions
    regions_sorted = sorted(countries, key=lambda c: (country_region_mapping.get(c, 'Other'), c))  # Sort by region, then alphabetically
    merged_with_regions_sorted = merged_with_regions[regions_sorted]

    return country_region_mapping, merged_with_regions_sorted

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
                               cv_folds=5, print_summary=True, return_scaler=False):
    """
    Trains and evaluates a regression model with options for scaling, cross-validation, and different model types.
    Optionally prints model summary and returns the trained model and/or scaler.

    Parameters:
    X (pandas DataFrame or numpy array): Independent variables.
    y (pandas DataFrame or numpy array): Target variable.
    model_type (str): Type of regression model ('linear', 'lasso', 'ridge'). Default: 'linear'.
    alphas (array): Alpha values for Lasso/Ridge. Default: logspace(-4,4,100).
    scale_data (bool): Whether to standardize features before training. Default: True.
    test_size (float): Test data proportion. Default: 0.3.
    random_state (int): Seed for reproducibility. Default: 66.
    cross_validate (bool): Perform cross-validation. Default: False.
    cv_folds (int): CV folds if cross_validate=True. Default: 5.
    print_summary (bool): Print model summary. Default: True.
    return_scaler (bool): Return the scaler object if scaling is performed. Default: False.

    Returns:
    dict: Dictionary containing training and testing metrics:
        - 'train_mse', 'train_r2', 'train_pearson', 'test_mse', 'test_r2', 'test_pearson'
    model: The trained regression model.
    scaler (optional): The fitted scaler if return_scaler=True and scale_data=True.
    """
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = None
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

    if print_summary:
        print("Training MSE:", metrics['train_mse'], 
              "Training R2:", metrics['train_r2'], 
              "Training Pearson Correlation:", metrics['train_pearson'])
        print("Testing MSE:", metrics['test_mse'], 
              "Testing R2:", metrics['test_r2'], 
              "Testing Pearson Correlation:", metrics['test_pearson'])

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

    # Return metrics and the model, plus scaler if requested and scaling was used
    if return_scaler and scale_data:
        return metrics, model, scaler
    else:
        return metrics, model


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

def calculate_trend(df):

    """
    Function for calculating and showing the  trends (slopes of linear regressions)
    
    """
    
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

    """
    Function for counting the beer styles appeared in the original BeerAdvocate and RateBeer databases for specific country
    
    """

    style_country_counts = {}

    unique_beer_styles = data['style'].unique()
    
    for style in unique_beer_styles:
    
        style_data = data[data['style'] == style]     # Filter data for the current beer style
        
        country_counts = style_data['user_location'].value_counts()       # Count the number of ratings by country for this style
        
        style_country_counts[style] = country_counts      # Store the result in the dictionary with the beer style as the key
    
    merged_df = pd.DataFrame(style_country_counts).transpose().fillna(0) # Convert the dictionary to a DataFrame

    return merged_df

def merge_data_with_counts_and_avg_ratings(data):
    """
    Function to aggregate beer style ratings and calculate both:
    1. Number of ratings for each beer style from each country.
    2. Average ratings for each beer style from each country.

    Parameters:
    - data: DataFrame containing columns `style`, `user_location`, `rating`.

    Returns:
    - merged_df: A pandas DataFrame where rows represent beer styles and columns represent countries,
                 with two sub-columns for each country:
                   - Number of ratings.
                   - Average ratings.
    """
    style_country_data = {}

    # Unique beer styles
    unique_beer_styles = data['style'].unique()

    for style in unique_beer_styles:

        style_data = data[data['style'] == style]                                      # Filter data for the current beer style
        
        country_counts = style_data['user_location'].value_counts()                    # Calculate the number of ratings for each country


        country_avg_ratings = style_data.groupby('user_location')['rating'].mean()     # Calculate the average rating for each country

        style_country_data[style] = {
            'count': country_counts,
            'avg_rating': country_avg_ratings
        }

    # Combine counts and average ratings into a multi index data frame
    count_df = pd.DataFrame({style: data['count'] for style, data in style_country_data.items()}).transpose().fillna(0)
    avg_rating_df = pd.DataFrame({style: data['avg_rating'] for style, data in style_country_data.items()}).transpose().fillna(0)

    # Add suffixes to differentiate columns
    count_df.columns = [f"{col}_count" for col in count_df.columns]
    avg_rating_df.columns = [f"{col}_avg_rating" for col in avg_rating_df.columns]

    # Merge the two data frame along columns
    merged_df = pd.concat([count_df, avg_rating_df], axis=1)

    return merged_df

def reduce_data_to_3D(dataframe):
    """
    Reduces a 4D dataset (count and average rating for each beer style) to 3D by calculating
    the product of count and average rating for each beer style.

    Returns:
    - pd.DataFrame: Reduced dataframe with one column per beer style (count * avg_rating).
    """
    # Extract all beer style names
    numeric_columns = [col for col in dataframe.columns if col.endswith('_count') or col.endswith('_avg_rating')]
    beer_styles = list(set([col.split('_')[0] for col in numeric_columns if '_count' in col]))

    # Initialize an empty DataFrame for reduced data
    reduced_data = pd.DataFrame()

    # Calculate count * avg_rating for each beer style
    for beer_style in beer_styles:
        count_col = f"{beer_style}_count"
        rating_col = f"{beer_style}_avg_rating"
        
        if count_col in dataframe.columns and rating_col in dataframe.columns:
            # Compute the product and store it in the reduced DataFrame
            reduced_data[beer_style] = dataframe[count_col] * dataframe[rating_col]

    return reduced_data

################################################# Plotting functions for making main code more concise #################################################

def plot_top_styles_over_time(agg_data, clusters, pca_transformed,countries_by_cluster,top_beer_styles_by_cluster):


    cluster_colors = {
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red',
        4: 'purple'
    }
    
    for cluster_id in sorted(pca_transformed['Cluster'].unique()):

        #  Extracting countries in the current cluster and filtering the actual data accordingly
        regions = countries_by_cluster[cluster_id][:]
        styles = list(top_beer_styles_by_cluster[cluster_id][0:5].keys())
    
        filtered_agg_data = agg_data[(agg_data['user_location'].isin(regions)) & (agg_data['style'].isin(styles))]     # Filter `agg_data` to include only the selected regions and beer styles
        filtered_agg_data = filtered_agg_data.groupby(['style', 'year']).agg(   #Data by Year, Region, and Beer Style
        avg_rating=('avg_rating', 'mean'),
        rating_count=('rating_count', 'size')   # Calculate the average rating per year, per region, and per beer style
    ).reset_index()
        plt.figure(figsize=(7, 5))
        for style in styles:
            subset = filtered_agg_data[(filtered_agg_data['style'] == style)]
            if len(subset) > 1:  # Ensure we have enough data points to plot a trend
                #color = color_mapping[region]  
                #marker = marker_mapping[style]     
                plt.plot(subset['year'], subset['avg_rating'], label=f'Cluster {cluster_id} - {style}',marker='^', linestyle='-',linewidth='2.0')
    
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.title("Regional Beer Style Preferences Over Time for Their Top 5 Styles")
        plt.legend(title="Region - Style", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.show()

def plot_regional_preferences(pca_transformed,subset_df_):

    # Define cluster colors from previous plot
    cluster_colors = {
        0: 'blue',
        1: 'orange',
        2: 'green',
        3: 'red',
        4: 'purple',
        5: 'brown',
        6: 'pink'
    }
    
    top_beer_styles_by_cluster = {}
    countries_by_cluster = {}
    
    # Create a combined figure
    fig, axes = plt.subplots(1, max(pca_transformed['Cluster'].unique())+1, figsize=(20, 6), sharey=False)
    fig.suptitle("Top 5 Beer Styles in Each Region(Cluster)", fontsize=16)
    
    for cluster_id in sorted(pca_transformed['Cluster'].unique()):
        #  Extracting countries in the current cluster and filtering the actual data accordingly
        cluster_countries = pca_transformed.loc[pca_transformed['Cluster'] == cluster_id, 'Country'].str.strip().tolist()
        countries = [country for country in cluster_countries if country in subset_df_.columns] 
        cluster_data = subset_df_[countries]
        
        cluster_data['average_number_of_ratings'] = cluster_data.mean(axis=1)                                # Calculate the average number of ratings for each beer style across countries
        
       
        top_5_beer_styles = cluster_data['average_number_of_ratings'].sort_values(ascending=False).head(5)   # Sort the beer styles by average rating in descending order and select the top 5
        
        top_beer_styles_by_cluster[cluster_id] = top_5_beer_styles
        countries_by_cluster[cluster_id] = cluster_countries
    
        top_5_beer_styles.plot(
            kind='bar',
            ax=axes[cluster_id],
            color=cluster_colors[cluster_id],
            alpha=0.7
        )
        axes[cluster_id].set_title(f"Cluster {cluster_id}")
        axes[cluster_id].set_xlabel("Beer Styles")
        axes[cluster_id].set_ylabel("Average Number of Rating" if cluster_id == 0 else "")
        axes[cluster_id].set_xticklabels(top_5_beer_styles.index, rotation=45, ha='right')
        axes[cluster_id].grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return top_beer_styles_by_cluster,countries_by_cluster

def plot_pca_loadings(loadingss,pc1_corr,pc2_corr):
    
    """
    Function for plotting the PCA loadings
    
    """
    
    loadingss['PC1'].sort_values(ascending=False).head(45).plot(kind='bar', color='blue', alpha=0.7, label='PC1')
    loadingss['PC2'].sort_values(ascending=False).head(45).plot(kind='bar', color='orange', alpha=0.7, label='PC2')
    
    
    plt.title('Top 45 Beer Styles Contributing to Principal Components\n'
              f'PC1 Correlation with Total Ratings: {pc1_corr:.2f}, '
              f'PC2 Correlation with Total Ratings: {pc2_corr:.2f}')
    plt.xlabel('Beer Styles')
    plt.ylabel('Loading (Contribution)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_PCA_2D_with_loadings(subset_df,title):

    
    """
    Adjusted PCA plot function with the effect of number of ratings
    
    """
    log_transformed_data = np.log10(subset_df.values + 1)           # Log-transform the data to handle large ranges (Adding 1 to avoid log(0)) 

    # Perform PCA (countries as rows, beer styles as columns)
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(log_transformed_data.T)     # Transpose for countries

    # Extract PCA loadings (beer style contributions)
    beer_styles = subset_df.index
    pca_loadings = pca.components_                                 # Shape: (n_components, n_features)
    loadings_df = pd.DataFrame(pca_loadings.T, index=beer_styles, columns=['PC1', 'PC2'])

    total_ratings = subset_df.sum(axis=1)                          # Total number of ratings per beer style

    # Calculate the correlation between the number of ratings and PCA loadings
    loadings_df['Total Ratings'] = total_ratings
    pc1_corr = loadings_df[['PC1', 'Total Ratings']].corr().iloc[0, 1]
    pc2_corr = loadings_df[['PC2', 'Total Ratings']].corr().iloc[0, 1]

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        pca_transformed[:, 0], pca_transformed[:, 1],
        c='blue', alpha=0.6, edgecolor='k', s=100
    )

    # Annotate countries on the scatter plot
    for i, country in enumerate(subset_df.columns):
        plt.annotate(country, (pca_transformed[i, 0], pca_transformed[i, 1]), fontsize=9, alpha=0.8)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid(alpha=0.3)
    
    plt.figure(figsize=(10, 6))
    plot_pca_loadings(loadings_df,pc1_corr,pc2_corr)

    plt.tight_layout()
    plt.show()
    
    return pca_transformed,loadings_df
    
def plot_clusters(pca_transformed):

    """
    Function for plotting the clusters
    
    """
    
    for cluster in sorted(pca_transformed['Cluster'].unique()):                   # Cluster labels
        cluster_data = pca_transformed[pca_transformed['Cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f"Cluster {cluster}", s=100)
    
    # Annotate countries
    for _, row in pca_transformed.iterrows():
        plt.annotate(row['Country'], (row['PC1'], row['PC2']), fontsize=9)
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Clustering on PCA-Transformed Data")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def silhoutte_inertia_plotting(cluster_range,pca_componentss):

    """
    Function for plotting the silhoutte widths and inertia scores
    
    """
    
    silhouette_scores = [] 
    inertia_values = []
    for n_clusters in cluster_range:                                            # Iterate through each number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_componentss)
        
        silhouette_avg = silhouette_score(pca_componentss, cluster_labels)      # Compute silhouette score and inertia
        inertia = kmeans.inertia_
        silhouette_scores.append(silhouette_avg)
        inertia_values.append(inertia)
    
    plt.figure(figsize=(9, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, silhouette_scores, marker='o', label='Silhouette Score')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Width Analysis")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, inertia_values, marker='o', label='Inertia (Elbow Method)', color='orange')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
      
def plot_over_time(agg_data,regions,styles):

    """
    Function for plotting the rating scores over time for chosen beer style in different countries
    
    """

    filtered_agg_data = agg_data[(agg_data['user_location'].isin(regions)) & (agg_data['style'].isin(styles))]     # Filter `agg_data` to include only the selected regions and beer styles

    color_mapping = {regions[0]: 'darkblue',regions[1]: 'red',regions[2]: 'magenta'}
    marker_mapping = {styles[0]: 'o',styles[1]: '^' }
    
    plt.figure(figsize=(7, 5))
    for region in regions:
        for style in styles:
            subset = filtered_agg_data[(filtered_agg_data['user_location'] == region) & (filtered_agg_data['style'] == style)]
            if len(subset) > 1:  # Ensure we have enough data points to plot a trend
                color = color_mapping[region]  
                marker = marker_mapping[style]     
                plt.plot(subset['year'], subset['avg_rating'], marker=marker, label=f'{region} - {style}', color=color, linestyle='-')
    
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.title("Regional Beer Style Preferences Over Time for American IPA and Pale Lager")
    plt.legend(title="Region - Style", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.show()
    
def plot_3D_scatter(subset_df,title):

    """
    Function for plotting the rating scores over time for chosen beer style in different countries
    
    """

    x_labels = subset_df.index                            # Beer styles on the x-axis
    y_labels = subset_df.columns                          # Countries on the y-axis
    x_pos, y_pos = np.meshgrid(range(len(x_labels)), range(len(y_labels)), indexing="ij")
    z_values = subset_df.values                           # Number of ratings on the z-axis
    
    x_pos_flat = x_pos.flatten()
    y_pos_flat = y_pos.flatten()
    z_values_log_flat = np.log10(z_values.flatten() + 1)  # Log scale to handle large ranges
    
    # Scatter plotting
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)
    scatter = ax.scatter(x_pos_flat, y_pos_flat, z_values_log_flat, c=z_values_log_flat, cmap='viridis', s=50)
    ax.set_xlabel('Beer Styles', labelpad=20)
    ax.set_ylabel('Countries', labelpad=20)
    ax.set_zlabel('Log(Number of Ratings)', labelpad=10)
    ax.set_title(title)
    
    ax.set_xticks(np.arange(0, len(x_labels), 5))
    ax.set_xticklabels(x_labels[::5], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(0, len(y_labels), 5))
    ax.set_yticklabels(y_labels[::5], fontsize=8)
    
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Log(Number of Ratings)')
    
    plt.show()

def plot_PCA_2D(subset_df):

    """
    Function for plotting 2D PCA of the given 3D data without implementing logarithmic scaling
    
    """
    # Convert the DataFrame into a matrix suitable for PCA
    data_for_pca = subset_df.values.T                         # Transpose: countries as rows, beer styles as columns

    # Perform PCA
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(data_for_pca)

    # Plot the transformed data
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c='blue', alpha=0.6, edgecolor='k', s=100)

    for i, country in enumerate(subset_df.columns):
        plt.annotate(country, (pca_transformed[i, 0], pca_transformed[i, 1]), fontsize=9, alpha=0.8)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Projection of European Countries Based on Beer Ratings')
    plt.grid(alpha=0.3)
    plt.show()
def top_10_predicted(top_10_styles,loc):
    
    """
    Function for plotting the top 10 beer style for seasonal trends
    
    """

    plt.figure(figsize=(12, 6))
    for style in top_10_styles.index:
        plt.plot(top_10_styles.columns, top_10_styles.loc[style], label=style)
    
    plt.title(f"Seasonal Trends in Predicted Ratings: {loc} 10 Beer Styles")
    plt.xlabel("Season")
    plt.ylabel("Predicted Rating")
    plt.legend(title="Beer Style", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
  
def plot_hist(attributes,ba_ratings_loc_filtered_no_missing,name):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, col in enumerate(attributes):
        sns.histplot(ba_ratings_loc_filtered_no_missing[col],
                     bins=int((ba_ratings_loc_filtered_no_missing[col].max() - ba_ratings_loc_filtered_no_missing[col].min()) / 0.05),
                     kde=False, ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(f"Distribution of {col} in {name}")
        axes[i // 2, i % 2].set_xlabel(col)
        axes[i // 2, i % 2].set_ylabel("Rating frequence")
        axes[i // 2, i % 2].grid(True, linestyle='--', alpha=0.7)  
    plt.tight_layout()
    plt.show()
  
def top_10_barh(data_x,data_y,title,color_,name):

    plt.figure(figsize=(12, 6))
    plt.barh(data_x, data_y, color=color_)
    plt.title(f"{title} 10 Countries by Average Rating - {name}")
    plt.xlabel("Average Rating")
    plt.ylabel("Country")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
def top_10_plots(ba_top_10_user_locations,rb_top_10_user_locations,title):

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.barplot(x=ba_top_10_user_locations.values, y=ba_top_10_user_locations.index, ax=axes[0])
    axes[0].set_title(f"Top 10 {title} in BeerAdvocate")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel(title)
    axes[0].set_xscale("log")
    
    sns.barplot(x=rb_top_10_user_locations.values, y=rb_top_10_user_locations.index, ax=axes[1])
    axes[1].set_title(f"Top 10 {title} in RateBeer")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel(title)
    axes[1].set_xscale("log")
    
    plt.tight_layout()
    plt.show()


def fit_logistic_regression_multi_output(X, Y, scale_data=True):
    """
    Fits multiple logistic regression models to the provided features and one-hot encoded target variable.

    parameters:
    X (pandas.DataFrame): feature matrix
    Y (pandas.DataFrame): target variable matrix (one-hot encoded)
    scale_data (bool): standardize the features before fitting the model or not

    returns:
    models (dict): dictionary of the fitted logistic regression models, keyed by target category
    propensity_scores (pandas.DataFrame): computed propensity scores for each instance and category
    """
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    models = {}
    propensity_scores = pd.DataFrame(index=Y.index)

    # fit a logistic regression model for each country (column in Y)
    for country in Y.columns:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, Y[country])
        models[country] = model
        propensity_scores[country] = model.predict_proba(X_scaled)[:, 1]  # probability of being from 'country'

    return models, propensity_scores



def match_reviews_propensity_score(data, score_col='propensity_score', location_col='user_location', max_diff=0.05):
    """
    Match reviews based on their propensity scores.
    
    Each review (row in data) should be matched to another review:
    - with a propensity score difference smaller than max_diff
    - with a different user_location
    - if multiple possible matches are found, choose the one with the smallest difference
    - if no suitable match is found, discard the review
    - each review can be paired only once

    parameters:
    data (pandas.DataFrame): dataframe containing at least two columns: user_location and propensity_score, where each row corresponds to a review
    score_col (str): name of the column containing the propensity scores
    location_col (str): name of the column containing the value of user_location
    max_diff (float): maximum allowed difference in propensity scores for matching

    returns:
    matched_df (pd.DataFrame): dataframe of matched pairs, with the columns from the original dataframe
    and matched rows; unmatched reviews are not included
    """

    # sort the data by propensity score for efficient searching
    data_sorted = data.sort_values(by=score_col).reset_index(drop=True)

    scores = data_sorted[score_col].values
    locations = data_sorted[location_col].values
    n = len(data_sorted)

    used = set() # keeps track of reviews that have already been matched
    matched_rows = []

    # iterate over each review and try to find the closest match
    for i in range(n):
        if i in used:
            continue # skip review if it's already matched
        
        current_score = scores[i]
        current_loc = locations[i]

        # use binary search to find the position where current_score would be inserted to maintain the sorted order, 
        # giving you index close to where similar scores are located
        pos = bisect_left(scores, current_score)

        best_match = None
        best_diff = max_diff

        # move left (towards lower scores) from pos to find suitable matches within max_diff of current_score,
        # updating best_match if a closer match is found
        j = pos - 1
        while j >= 0 and (current_score - scores[j]) <= max_diff:
            if j not in used and locations[j] != current_loc:
                diff = abs(current_score - scores[j])
                if diff < best_diff:
                    best_diff = diff
                    best_match = j
            j -= 1

        # move right (towards higher scores) from pos, again looking for close matches and updating best_match
        # if a better candidate is found
        j = pos
        while j < n and (scores[j] - current_score) <= max_diff:
            if j not in used and j != i and locations[j] != current_loc:
                diff = abs(current_score - scores[j])
                if diff < best_diff:
                    best_diff = diff
                    best_match = j
            j += 1

        # if a suitable match is found, record it
        if best_match is not None:
            used.add(i)
            used.add(best_match)
            # combine the two matched rows into one
            matched_pair = pd.concat(
                [data_sorted.iloc[i], data_sorted.iloc[best_match].add_suffix('_matched')],
                axis=0
            )
            matched_rows.append(matched_pair)

    if len(matched_rows) == 0:
        return pd.DataFrame()

    matched_df = pd.DataFrame(matched_rows)
    return matched_df


def standardize_pair(row):
    """
    Given a row with 'user_location', 'user_location_matched', 'rating', and 'rating_matched' columns, 
    ensures a consistent ordering of country pairs (alphabetically sorted) and adjusts the ratings accordingly
    so that the first listed country always corresponds to 'std_rating' and the second to 'std_rating_matched'.
    """
    loc1, loc2 = row['user_location'], row['user_location_matched']
    if loc1 > loc2:
        # swap countries and associated ratings
        return pd.Series({
            'user_location': loc2,
            'user_location_matched': loc1,
            'rating': row['rating_matched'],
            'rating_matched': row['rating']
        })
    else:
        # already in correct alphabetical order
        return pd.Series({
            'user_location': loc1,
            'user_location_matched': loc2,
            'rating': row['rating'],
            'rating_matched': row['rating_matched']
        })


def compare_countries_bidirectional(matched_data, test_func=ttest_rel):
    """
    Compares paired rating differences between all pairs of user locations, ensuring
    that (A, B) and (B, A) are grouped together and that the rating difference direction is consistent.

    parameters:
    matched_data (pandas.DataFrame): a dataFrame containing matched reviews
    (required columns: 'user_location', 'user_location_matched', 'rating', 'rating_matched')
    test_func (function): the statistical paired test to use, eg ttest_rel

    returns:
    results (pandas.DataFrame): a dataframe with one row per unique country pair (in alphabetical order)
    and columns for the test statistic, p-value, and sample size
    """

    df = matched_data.copy()
    df['user_location'] = df['user_location'].astype(str)
    df['user_location_matched'] = df['user_location_matched'].astype(str)

    # apply the standardization function row-wise
    standardized = df.apply(standardize_pair, axis=1)
    df = pd.concat([df.drop(columns=['user_location', 'user_location_matched', 'rating', 'rating_matched']), standardized], axis=1)

    # group by standardized country pairs
    grouped = df.groupby(['user_location', 'user_location_matched'], dropna=False)
    results = []

    for (loc1, loc2), group in grouped:
        if len(group) < 20:
            continue  # not enough data for a meaningful test

        # perform the paired test on rating vs rating_matched
        t_stat, p_value = test_func(group['rating'], group['rating_matched'])

        results.append({
            'country_1': loc1,
            'country_2': loc2,
            'test_stat': t_stat,
            'p_value': p_value,
            'n_pairs': len(group)
        })

    return pd.DataFrame(results)


def plot_pvalues(results_df, alpha=0.05, random_state=42):
    """
    Plots the p-values from the results dataframe for each country pair, adds a horizontal line at y=alpha, and labels only the p-values 
    smaller than alpha with their corresponding country pairs. Also shuffles the rows of the dataframe to avoid placing points 
    from the same first country consecutively, reducing label overlap.

    Additionally:
    - data points with p_value < alpha and test_stat > 0 are green
    - Data points with p_value < alpha and test_stat < 0 are red
    - other points are black

    parameters:
    results_df (pandas.DataFrame): dataframe returned by compare_countries_bidirectional
    (expected columns: ['country_1', 'country_2', 'test_stat', 'p_value'])
    alpha (float): significance level at which to draw the horizontal line (default 0.05)
    random_state (int): seed for reproducible shuffling

    returns:
    None; displays the plot
    """

    # shuffle the rows of the dataframe
    results_df = results_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # create a column that combines the country names for labeling
    results_df['pair_label'] = results_df['country_1'] + ' vs ' + results_df['country_2']

    # determine colors based on conditions
    colors = []
    for _, row in results_df.iterrows():
        if row['p_value'] < alpha:
            if row['test_stat'] > 0:
                colors.append('green')
            elif row['test_stat'] < 0:
                colors.append('red')
            else:
                # if test_stat == 0 and p_value > alpha
                colors.append('gray')
        else:
            colors.append('black')

    fig, ax = plt.subplots(figsize=(22, 8))

    # plot the p-values with assigned colors as a scatter plot
    ax.scatter(results_df.index, results_df['p_value'], color=colors, s=50)

    # draw a horizontal line at y=alpha
    ax.axhline(y=alpha, color='black', linestyle='--', linewidth=1)

    # label points with p-values < alpha
    for i, row in results_df.iterrows():
        if row['p_value'] < alpha:
            ax.text(i, row['p_value'], row['pair_label'], 
                    ha='center', va='bottom', rotation=90, fontsize=9)

    # create legend
    legend_elements = [
        Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=10, label='p < 0.05, test_stat > 0'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=10, label='p < 0.05, test_stat < 0'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=10, label='Other points'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1, label=f'Alpha = {alpha}')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlabel('Pair index')
    ax.set_ylabel('p-value')
    ax.set_title('p-values from test comparing the difference in final beer rating by country pair')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_top_10_pairs_boxplot(matched_data, results_df):
    """
    Plots a boxplot of rating differences for the top 10 country pairs (smallest p-values) from the results_df. 
    Each box represents the distribution of rating differences (rating - rating_matched) for one country pair.

    parameters:
    matched_data (pandas.DataFrame): contains matched reviews with columns:
    ['user_location', 'user_location_matched', 'rating', 'rating_matched']
    results_df (pandas.DataFrame): results of compare_countries_bidirectional, with columns:
    ['country_1', 'country_2', 'test_stat', 'p_value', 'n_pairs']

    the function:
    1) sorts results_df by p_value and selects the top 10
    2) standardizes matched_data so that each pair is represented consistently as (A, B) where A < B
    3) filters matched_data to include only those top 10 pairs
    4) plots the boxplot of rating differences
    """
    
    # 1) get the top 10 country pairs by smallest p-value
    top_10 = results_df.sort_values(by='p_value', ascending=True).head(10)
    
    # create a set of (country_1, country_2) tuples representing these pairs for easy filtering
    top_10_pairs = set(zip(top_10['country_1'], top_10['country_2']))

    # 2) standardize matched_data to ensure a consistent pair labeling
    # standardize_pair should return columns: user_location, user_location_matched, rating, rating_matched
    df = matched_data.copy()
    standardized = df.apply(standardize_pair, axis=1)
    df = pd.concat([df.drop(columns=['user_location', 'user_location_matched', 'rating', 'rating_matched']), standardized], axis=1)

    # create a pair_label column for filtering and plotting
    df['pair_label'] = df['user_location'] + ' vs ' + df['user_location_matched']

    # create a tuple column to match with top_10_pairs
    df['pair_tuple'] = list(zip(df['user_location'], df['user_location_matched']))

    # 3) filter the dataframe to only include top 10 pairs
    df_filtered = df[df['pair_tuple'].isin(top_10_pairs)].copy()

    # compute the rating difference within pairs
    df_filtered['rating_difference'] = df_filtered['rating'] - df_filtered['rating_matched']

    # order pairs by p-value
    pair_order = top_10.apply(lambda row: (row['country_1'], row['country_2']), axis=1).tolist()

    # 4) plot boxplot 
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x='pair_label', y='rating_difference', 
                order=[f"{a} vs {b}" for (a,b) in pair_order], color='orange')
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Distribution of rating differences for top 10 country pairs by p-value')
    plt.xlabel('Country pair')
    plt.ylabel('Rating difference (country_1 - country_2)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_seasonal_heatmap(predicted_table):
    """
    Plots a heatmap of predicted beer ratings across different seasons and beer styles.

    parameters:
    predicted_table (pandas.DataFrame): apivot table where rows are beer styles, columns are seasons, 
    and values are predicted ratings
        
    returns:
    None; displays the heatmap of predicted ratings
    """
    plt.figure(figsize=(14, 10))

    # sort styles by their average predicted rating across seasons
    sorted_styles = predicted_table.mean(axis=1).sort_values(ascending=False).index
    sorted_table = predicted_table.loc[sorted_styles]

    # create heatmap
    sns.heatmap(
        sorted_table,
        annot=False,
        cmap="coolwarm",
        cbar_kws={'label': 'Predicted rating'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title("Seasonal variations in predicted ratings by beer style", fontsize=18)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Beer style", fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plt.show()
    plt.close()

def plot_top_10_boxplot(df, custom_title= 'Distribution of rating differences (domestic - foreign) for countries with the highest mean difference'):
    """
    Plots a boxplot showing the distribution of (rating_domestic - rating_foreign) 
    for the top 10 user_location_domestic values with the largest mean differences.

    parameters:
    df (panadas.DataFrame): dataframe with columns 'user_location_domestic', 'rating_domestic', and 'rating_foreign'

    returns: None; displays a boxplot
    """
    # calculate rating differences
    df['rating_difference'] = df['rating_domestic'] - df['rating_foreign']

    # calculate mean differences for each user_location_domestic
    mean_differences = df.groupby('user_location_domestic')['rating_difference'].mean()

    # select the top 10 user_location_domestic with the largest mean differences in absolute value
    top_10_locations = mean_differences.abs().nlargest(10).index
    filtered_df = df[df['user_location_domestic'].isin(top_10_locations)]

    # plot the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=filtered_df, 
        x='user_location_domestic', 
        y='rating_difference', 
        order=top_10_locations
    )
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title(custom_title)
    plt.xlabel('Country')
    plt.ylabel('Rating difference (domestic - foreign)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()


def match_reviews(df, filter_column, group1_value, group2_value, join_columns, suffixes=('_group1', '_group2')):
    """
    Matches reviews from two groups in a given dataframe based on specified conditions.

    parameters:
    df (pandas.DataFrame): input dataframe containing the filter column and matching keys
    filter_column (str): column used to filter the two groups
    group1_value: value in filter_column representing the first group
    group2_value: value in filter_column representing the second group
    join_columns (list): list of columns to use for matching rows
    suffixes (tuple): suffixes for the matched columns (default: ('_group1', '_group2'))

    returns:
    matched_reviews (pandas.DataFrame): a dataframe with paired reviews from the two groups
    """
    # add a unique ID to each review
    df = df.reset_index() 
    df['review_id'] = df.index

    # filter for the two groups based on the filter_column
    group1_reviews = df[df[filter_column] == group1_value]
    group2_reviews = df[df[filter_column] == group2_value]

    # merge the two groups on the specified join columns
    matched_reviews = pd.merge(
        group1_reviews,
        group2_reviews,
        on=join_columns,
        suffixes=suffixes,
        how='inner'
    )

    # drop duplicate matches based on unique IDs
    matched_reviews = matched_reviews.drop_duplicates(subset=[f'review_id{suffixes[0]}'])
    matched_reviews = matched_reviews.drop_duplicates(subset=[f'review_id{suffixes[1]}'])

    return matched_reviews


def plot_rating_differences_histogram(data, diff_column, bins=30, color='orange', custom_title='Histogram of rating differences'):
    """
    Plots a histogram of rating differences with reference lines for zero and the mean.

    parameters:
    data (pandas.DataFrame): dataframe containing the rating differences
    diff_column (str): column name for the rating differences
    bins (int): number of bins for the histogram (default: 30)
    color (str): color of the histogram bars (default: 'orange')
    custom_title (str): custom title for the plot (default: 'Histogram of rating differences')

    Returns: None; displays the plot.
    """
    # calculate the mean rating difference
    mean_difference = data[diff_column].mean()

    # plot the histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(data[diff_column], bins=bins, color=color, kde=False)
    plt.axvline(x=0, color='black', linestyle='--', label='Reference (0)')  # zero line
    plt.axvline(x=mean_difference, color='blue', linestyle='-', label=f'Mean ({mean_difference:.2f})')  # mean line
    plt.title(custom_title)
    plt.xlabel('Rating Difference (Domestic - Foreign)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()
########################################################################################################################################################
  