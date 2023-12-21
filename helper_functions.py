import os
import shutil

from tempfile import NamedTemporaryFile
import pandas as pd

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


def create_ordered_dataframe(original_df, time_col, independent_feats, target_var):
    df = pd.DataFrame()
    df[time_col] = original_df[time_col]
    df[independent_feats] = original_df[independent_feats]
    df[target_var] = original_df[target_var]
    return df