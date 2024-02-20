import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def detect_null_values(df: pd.DataFrame):
    # Calculate the percentage of missing values in entire dataframe
    missing_values = df.isnull().sum()
    total_cells = np.product(df.shape)
    total_missing = missing_values.sum()
    return (total_missing / total_cells) * 100


def detect_inconsistent_types(df: pd.DataFrame):
    # Assume first column of dataframe has dates
    # Last column has target variable
    # Rest of the columns are features
    date_column = df.columns[0]
    target_column = df.columns[-1]
    feature_columns = df.columns[1:-1]
    date_column_type = df[date_column].dtype
    target_column_type = df[target_column].dtype
    feature_columns_type = df[feature_columns].dtypes
    inconsistent_date_type = date_column_type != 'datetime64[ns]'
    inconsistent_target_type = target_column_type != 'float64'
    # assign true if any of the feature columns are not of type float64
    inconsistent_feature_type = any(feature_columns_type != 'float64')

    return inconsistent_date_type, inconsistent_target_type, inconsistent_feature_type


def detect_duplicates(df: pd.DataFrame):
    # Calculate percentage of duplicate rows in entire dataframe
    total_rows = df.shape[0]
    duplicate_rows = df.duplicated().sum()
    return (duplicate_rows / total_rows) * 100


def detect_outliers(df: pd.DataFrame):
    # Assume last column is target variable
    target_column = df.columns[-1]
    # Calculate percentage of outliers in target variable
    q1 = df[target_column].quantile(0.25)
    q3 = df[target_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
    return (outliers.shape[0] / df.shape[0]) * 100


def check_similarity(vals_list, range_threshold, std_threshold):
    range_vals = max(vals_list) - min(vals_list)
    std_dev = np.std(vals_list)

    return ((range_vals < range_threshold) and (std_dev < std_threshold))


def detect_need_for_scaling(df: pd.DataFrame):
    # features_min = df.iloc[:, 1:-1].min(axis=0)
    # features_max = df.iloc[:, 1:-1].max(axis=0)
    # features_range = features_max - features_min

    # range_threshold = 1
    # std_threshold = 1

    # similar_min = check_similarity(features_min.tolist(), range_threshold, std_threshold)
    # similar_max = check_similarity(features_max.tolist(), range_threshold, std_threshold)
    # similar_range = check_similarity(features_range.tolist(), range_threshold, std_threshold)

    # return (similar_min and similar_max and similar_range)

    features_min = df.iloc[:, 1:-1].min(axis=None)
    features_max = df.iloc[:, 1:-1].max(axis=None)
    features_range = features_max - features_min

    range_threshold = 1

    if (features_range <= range_threshold):
        return False
    else:
        return True


def get_preprocessing_needs_table(df):
    perc_null_values = round(detect_null_values(df), 2)
    inconsistency_types = detect_inconsistent_types(df)
    inconsistency = f"date: {inconsistency_types[0]}, target: {inconsistency_types[1]}, features: {inconsistency_types[2]}"
    # inconsistency = inconsistency_types[0] or inconsistency_types[1] or inconsistency_types[2]
    perc_duplicates = round(detect_duplicates(df), 2)
    perc_outliers = round(detect_outliers(df), 2)
    needs_scaling = detect_need_for_scaling(df)

    table = pd.DataFrame({
        'Type': ['Percentage of Null Values', 'Inconsistent Types', 'Percentage of Duplicates', 'Percentage of Outliers', 'Need for Scaling'],
        'Value': [perc_null_values, inconsistency, perc_duplicates, perc_outliers, needs_scaling]
    })
    
    return table


#############################################################################


def fill_null_values(df: pd.DataFrame):
    # Fill null values in each column with linear interpolation
    df = df.interpolate(method='linear', axis=0)
    return df


def fix_inconsistent_types(df: pd.DataFrame):
    # Assume first column of dataframe has dates
    # Last column has target variable
    # Rest of the columns are features
    date_column = df.columns[0]
    target_column = df.columns[-1]
    feature_columns = df.columns[1:-1]

    # Convert date column to datetime64 type
    df[date_column] = pd.to_datetime(df[date_column])

    # Convert target column to float64 type
    df[target_column] = df[target_column].astype('float64')

    # Convert feature columns to float64 type
    df[feature_columns] = df[feature_columns].astype('float64')

    return df


def remove_duplicates(df: pd.DataFrame):
    # Remove duplicate rows from dataframe
    df = df.drop_duplicates(subset=df.columns[0], keep='first')
    return df


def remove_outliers(df: pd.DataFrame):
    # Assume last column is target variable
    target_column = df.columns[-1]
    # Remove outliers from target variable
    q1 = df[target_column].quantile(0.25)
    q3 = df[target_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df = df[(df[target_column] > lower_bound) & (df[target_column] < upper_bound)]
    return df


def scale_features(df: pd.DataFrame):
    # Scale features using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
    return df


#############################################################################

def plot_data(df: pd.DataFrame):
    # Assume first column of dataframe has dates
    # Last column has target variable
    # Rest of the columns are features
    try:
        date_column = df.columns[0]
        target_column = df.columns[-1]
        feature_columns = df.columns[1:-1]

        dates = df[date_column].astype(str)
        xticks = dates.iloc[::len(dates)//10]

        total_plots = len(feature_columns) + 1
        ax_num = math.ceil(math.sqrt(total_plots))

        if total_plots <= ax_num * (ax_num - 1):
            ncols = ax_num
            nrows = math.ceil(total_plots / ncols)
        else:
            nrows = ax_num
            ncols = math.ceil(total_plots / nrows)

        fig, ax = plt.subplots(nrows, ncols, figsize=(10, 8))

        # Plot each feature column against date column
        i, j = 0, 0
        for feature in feature_columns:
            ax[i][j].plot(df[date_column], df[feature], label=feature)
            ax[i][j].set_xlabel("Dates")
            ax[i][j].set_ylabel(feature)
            ax[i][j].set_xticks(xticks)
            ax[i][j].set_xticklabels(xticks, rotation=90)

            if j == ax_num - 1:
                i += 1
                j = 0
            else:
                j += 1
        
        # ax[0].legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
        # ax[0].set_ylabel('Feature Variables')

        # ax[0].set_title('Data Visualization')
        # ax[0].set_xticks([])
        # Plot target column against date column
        ax[i][j].plot(df[date_column], df[target_column], label=target_column, color='black')
        ax[i][j].set_xlabel("Dates")
        ax[i][j].set_xticks(xticks)
        ax[i][j].set_xticklabels(xticks, rotation=90)
        ax[i][j].set_ylabel('Target Variable')

        fig.suptitle('Data Visualization', fontsize=16)
        plt.tight_layout()        
        return fig, None
    except Exception as e:
        return None, e
