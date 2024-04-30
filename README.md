# Bluegill: An Autonomic Machine Learning Platform
 A dashboard created to help train classification and regression models using machine learning and deep learning and evaluate them using explainable AI methods

 ## Steps for setting up Bluegill:

1. **Setting up the Conda environment**

   If you do not have Anaconda on your system, follow the instructions on [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install it based on your OS.

   Run the following command in the Anaconda prompt to create a new environment for the web app:
   ```bash
   $ conda create -n bluegill-venv python=3.11.5
   ```

2. **Installing the required libraries**
   
   Activate the environment by running the following command:
   ```bash
   $ conda activate bluegill-venv
   ```

   Now navigate to the source code directory in the Anaconda prompt by running the following command:
   ```bash
   $ cd path_to_directory
   ```
   (after replacing "path_to_directory" with the actual absolute path to the Folder with the source code)

   Install the required libraries by running the command:
   ```bash
   $ pip install -r requirements.txt
   ```
   This should take a couple of minutes

3. **Starting Bluegill**
   
   Start the Web App by running the command:
   ```bash
   $ streamlit run app.py
   ```
4. **Terminating the Web App**
   
   In the Anaconda prompt, enter "Ctrl+C" to terminate the app.
