import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from scipy import stats


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
    num_outliers = 0
    for col in df.columns[1:-1]:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            num_outliers += outliers.shape[0]
        except:
            continue
    
    return (num_outliers / df.shape[0]) * 100
    # # Assume last column is target variable
    # target_column = df.columns[-1]
    # # Calculate percentage of outliers in target variable
    # q1 = df[target_column].quantile(0.25)
    # q3 = df[target_column].quantile(0.75)
    # iqr = q3 - q1
    # lower_bound = q1 - (1.5 * iqr)
    # upper_bound = q3 + (1.5 * iqr)
    # outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
    # return (outliers.shape[0] / df.shape[0]) * 100


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
    min_val, max_val = np.inf, -np.inf
    for col in df.columns[1:-1]:
        try:
            min_val = min(min_val, df[col].min())
            max_val = max(max_val, df[col].max())
        except:
            continue

    # features_min = df.iloc[:, 1:-1].min(axis=None)
    # features_max = df.iloc[:, 1:-1].max(axis=None)
    features_min = min_val
    features_max = max_val
    features_range = features_max - features_min

    range_threshold = 1

    if (features_range <= range_threshold):
        return False
    else:
        return True


def detect_data_type(df: pd.DataFrame):
    target_column = df.columns[-1]

    if df[target_column].nunique() > 10:
        return "Data suitable for Regression", None
    else:
        class_counts = df[target_column].value_counts()
        imbalance_threshold = 0.5  # You can adjust this threshold based on your data
        minority_class_count = class_counts.min()
        majority_class_count = class_counts.max()
        imbalance_ratio = minority_class_count / majority_class_count
        if imbalance_ratio > imbalance_threshold:
            return "Data suitable for Classification with insignificant class imbalance", class_counts
        else:
            return "Data suitable for Classification but needs to handle class imbalance", class_counts


def detect_periodicity(df: pd.DataFrame):
    try:
        # Assume first column of dataframe has dates
        date_column = df.columns[0]
        # Calculate the frequency of the time series data
        time_diffs = df[date_column].diff().dropna()
        most_common_periodicity = time_diffs.mode()[0]
        min_periodicity = time_diffs.min()
        # time_diffs = time_diffs.dt.total_seconds()
        # time_diffs = time_diffs[time_diffs > 0]
        # time_diffs = time_diffs.value_counts()
        # time_diffs = time_diffs[time_diffs > 1]
        most_common_day_per = most_common_periodicity.days
        min_day_per = min_periodicity.days
        return f"Most common periodicity: {most_common_day_per} days, Minimum periodicity: {min_day_per} days"
    except Exception as e:
        return e


def detect_columns_for_encoding(df: pd.DataFrame):
    # Assume first column of dataframe has dates
    # Last column has target variable
    # Rest of the columns are features
    feature_columns = df.columns[1:-1]

    # Check if any of the feature columns are of type object
    columns_for_encoding = []
    for col in feature_columns:
        if df[col].dtype == 'object' or df[col].dtype == 'str':
            columns_for_encoding.append(col)
    return columns_for_encoding


def get_preprocessing_needs_table(df):
    perc_null_values = round(detect_null_values(df), 2)
    inconsistency_types = detect_inconsistent_types(df)
    inconsistency = f"date: {inconsistency_types[0]}, target: {inconsistency_types[1]}, features: {inconsistency_types[2]}"
    # inconsistency = inconsistency_types[0] or inconsistency_types[1] or inconsistency_types[2]
    perc_duplicates = round(detect_duplicates(df), 2)
    perc_outliers = round(detect_outliers(df), 2)
    needs_scaling = detect_need_for_scaling(df)
    data_type = detect_data_type(df)
    periodicity = detect_periodicity(df)
    columns_for_encoding = detect_columns_for_encoding(df)

    null_contribution = (100 - perc_null_values) / 100
    inconsistency_contribution = (3 - sum(inconsistency_types)) / 3
    duplicates_contribution = (100 - perc_duplicates) / 100
    outliers_contribution = (100 - perc_outliers) / 100
    scaling_contribution = 0 if needs_scaling else 1
    data_type_contribution = 0 if data_type[0] == "Data suitable for Classification but needs to handle class imbalance" else 1

    progress = (null_contribution + inconsistency_contribution + duplicates_contribution + outliers_contribution + scaling_contribution + data_type_contribution) / 6

    table = pd.DataFrame({
        'Type': ['Percentage of Null Values', 'Inconsistent Types', 'Percentage of Duplicates', 'Percentage of Outliers', 'Need for Scaling', 'Data Type', 'Periodicity', 'Columns for Encoding'],
        'Value': [perc_null_values, inconsistency, perc_duplicates, perc_outliers, needs_scaling, data_type, periodicity, columns_for_encoding]
    })
    
    return table, progress


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
    for feat_col in feature_columns:
        try:
            df[feat_col] = df[feat_col].astype('float64')
        except:
            continue
    # df[feature_columns] = df[feature_columns].astype('float64')

    return df


def remove_duplicates(df: pd.DataFrame):
    # Remove duplicate rows from dataframe
    df = df.drop_duplicates(subset=df.columns[0], keep='first')
    return df


def remove_outliers(df: pd.DataFrame):
    for col in df.columns[1:-1]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    # # Assume last column is target variable
    # target_column = df.columns[-1]
    # # Remove outliers from target variable
    # q1 = df[target_column].quantile(0.25)
    # q3 = df[target_column].quantile(0.75)
    # iqr = q3 - q1
    # lower_bound = q1 - (1.5 * iqr)
    # upper_bound = q3 + (1.5 * iqr)
    # df = df[(df[target_column] > lower_bound) & (df[target_column] < upper_bound)]
    return df


def scale_features(df: pd.DataFrame):
    # Scale features using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
    return df


def fix_class_imbalance(df: pd.DataFrame):
    date_column = df.columns[0]
    # Assume last column is target variable
    target_column = df.columns[-1]
    # Handle class imbalance using SMOTE
    # over = SMOTE(sampling_strategy=0.1)
    # under = RandomUnderSampler(random_state=42)
    # steps = [('o', over), ('u', under)]
    # steps = [('u', under)]
    # pipeline = Pipeline(steps=steps)

    X = df.iloc[:, :-1]
    y = df[target_column]
    
    # X_und, y_und = under.fit_resample(X, y)
    # new_df = pd.concat([X_und, y_und], axis=1)

    # get value counts of target variable
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    minority_class_count = class_counts.min()

    resampled_dfs = []
    for class_val in class_counts.index:
        resampled_dfs.append(df[df[target_column] == class_val].sample(minority_class_count, replace=False, random_state=42))
    new_df = pd.concat(resampled_dfs).sort_values(by=date_column)
        
    return new_df


def interpolate_data(df:pd.DataFrame, freq:str):
    # Assuming first column of df has dates in MM/DD/YYYY
    # and the last column has the target values.
    dates_name = df.columns[0]
    # target_name = df.columns[1:]

    df[dates_name] = pd.to_datetime(df[dates_name], dayfirst=False)
    date_range = pd.date_range(start=df[dates_name].min(), end=df[dates_name].max(), freq=freq)
    complete_df = pd.DataFrame({dates_name:date_range})

    result_df = pd.merge(complete_df, df, on=dates_name, how='left')
    final_df = result_df.copy(deep=True)

    final_df.interpolate(inplace=True)
    return final_df


def boxcox_transform(df: pd.DataFrame):
    # Assume first column of dataframe has dates
    # Last column has target variable
    # Rest of the columns are features
    feature_columns = df.columns[1:-1]
    feat_lmda = {}
    for col in feature_columns:
        df[col].replace(0, 0.001, inplace=True)
        df[col].replace(0., 0.001, inplace=True)   
        try:
            df[col], lmbda = stats.boxcox(df[col])
            feat_lmda[col] = lmbda
        except:
            feat_lmda[col] = None
    return df, feat_lmda


def encode_categorical_features(df: pd.DataFrame, cols: list, type: str):
    if type == 'one_hot':
        encoder_dict = {}        
        for col in cols:
            encoder = OneHotEncoder()
            enc_vals = encoder.fit_transform(df[col].values.reshape(-1, 1)).toarray()
            encoder_dict[col] = encoder
            vals = []
            for i in range(df.shape[0]):
                vals.append(enc_vals[i])
            df[col] = vals
        return df, encoder_dict
    elif type == 'label':
        for col in cols:
            df[col] = df[col].astype('category').cat.codes
        return df


def get_correlation_matrix(df: pd.DataFrame, method: str = 'pearson'):
    # Assume first column of dataframe has dates
    # Last column has target variable
    # Rest of the columns are features
    feature_columns = df.columns[1:-1]
    try:
        corr_matrix = df[feature_columns].corr(method=method)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        return fig, None
    except Exception as e:
        return None, e


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
            try:
                ax[i][j].plot(df[date_column], df[feature], label=feature)
            except:
                # Write as text
                ax[i][j].text(0.5, 0.5, f"Cannot plot {feature}", horizontalalignment='center', verticalalignment='center', transform=ax[i][j].transAxes)
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
