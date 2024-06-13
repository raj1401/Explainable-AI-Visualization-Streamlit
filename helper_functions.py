import os
import shutil

from tempfile import NamedTemporaryFile
import pandas as pd
import streamlit as sl

def delete_folder_contents(folder_path):
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        print(f"Contents of {folder_path} deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")


@sl.cache_data
def data_file_loader(data_file, temp_dir):
    if data_file is not None:
        try:
            with NamedTemporaryFile(mode='wb', suffix=".csv", dir=temp_dir, delete=False) as f:
                f.write(data_file.read())
            with open(f.name, 'r') as file:
                df = pd.read_csv(file)
            return df
        except Exception as e:
            return None
    else:
        return None

@sl.cache_data
def create_ordered_dataframe(original_df, time_col, independent_feats, target_var, binarize_threshold=None):
    df = pd.DataFrame()
    df[time_col] = original_df[time_col]
    df[independent_feats] = original_df[independent_feats]
    if binarize_threshold is not None:
        df[target_var] = original_df[target_var].apply(lambda x: 1 if x >= binarize_threshold else 0).astype('float64')
    else:
        df[target_var] = original_df[target_var]
    return df


def multi_time_series_loader(data_file, temp_dir):
    if data_file is not None:
        try:
            with NamedTemporaryFile(mode='wb', suffix=".csv", dir=temp_dir, delete=False) as f:
                f.write(data_file.read())
            with open(f.name, 'r') as file:
                df = pd.read_csv(file)
            series_list = []
            for i in range(len(df.columns)):
                series_list.append(df.iloc[:,i])
            print(series_list[0])
            return series_list
        except Exception as e:
            return None
    else:
        return None