import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, date, time
from itertools import product
import seaborn as sns
from scipy import stats
import scipy.stats as st
import glob
import warnings
import statsmodels.api as sm  

from wordcloud import WordCloud
from PIL import Image
from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


from src.utils.data_utils import advanced_linear_regression
from src.utils.data_utils import assign_experience_level
from src.utils.data_utils import clean_location_column
from src.utils.data_utils import get_season
from src.utils.data_utils import merge_data_
from src.utils.data_utils import calculate_trend
from src.utils.data_utils import plot_coefficients
from src.utils.data_utils import top_10_plots
from src.utils.data_utils import top_10_barh
from src.utils.data_utils import plot_3D_scatter
from src.utils.data_utils import plot_over_time
from src.utils.data_utils import top_10_predicted
from src.utils.data_utils import plot_hist
from src.utils.data_utils import region_mapping
from src.utils.data_utils import generate_wordcloud_from_series
from src.utils.data_utils import plot_pca_loadings
from src.utils.data_utils import plot_PCA_2D_with_loadings
from src.utils.data_utils import plot_clusters
from src.utils.data_utils import silhoutte_inertia_plotting
from src.utils.data_utils import plot_PCA_2D