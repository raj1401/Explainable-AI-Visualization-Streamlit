import os
import shutil

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