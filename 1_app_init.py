import streamlit as st
from helper_functions import data_file_loader, create_ordered_dataframe


# ---- GLOBAL VARIABLES ---- #
TEMP_DIR = "temp_data"

# ---- SESSION STATE VARIABLES ---- #
if 'classification_models' not in st.session_state:
    st.session_state.classification_models = ["Random Forest Classifier", "Logistic Regression", "KNN Classifier", "SVM Classifier"]

if 'regression_models' not in st.session_state:
    st.session_state.regression_models = ["Random Forest Regressor", "Linear Regression", "KNN Regressor", "SVM Regressor"]

if 'all_models' not in st.session_state:
    st.session_state.all_models = st.session_state.classification_models + st.session_state.regression_models

if 'available_feat_select_algos' not in st.session_state:
    st.session_state.available_feat_select_algos = ["SHAP", "Recursive Feature Elimination", "Boruta"]

if 'original_df' not in st.session_state:
    st.session_state.original_df = None

if 'all_col_names' not in st.session_state:
    st.session_state.all_col_names = None

if 'time_col' not in st.session_state:
    st.session_state.time_col = None

if 'independent_feats' not in st.session_state:
    st.session_state.independent_feats = None

if 'target_var' not in st.session_state:
    st.session_state.target_var = None

if 'search_results' not in st.session_state:
    st.session_state.search_results = {}

if 'final_params' not in st.session_state:
    st.session_state.final_params = {}

if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

if 'feats_selected_df' not in st.session_state:
    st.session_state.feats_selected_df = None

if 'final_model' not in st.session_state:
    st.session_state.final_model = None

if 'model_on_selected_feats' not in st.session_state:
    st.session_state.model_on_selected_feats = None

if 'trigger_training_on_new_feats' not in st.session_state:
    st.session_state.trigger_training_on_new_feats = False

if 'TRAIN_TEST_RANDOM_STATE' not in st.session_state:
    st.session_state.TRAIN_TEST_RANDOM_STATE = None

if 'TEST_FRACTION' not in st.session_state:
    st.session_state.TEST_FRACTION = None

if 'LSTM_Model' not in st.session_state:
    st.session_state.LSTM_Model = None

if 'box_cox_dict' not in st.session_state:
    st.session_state.box_cox_dict = None


# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="Explainable AI", layout='wide')

# ---- TITLE ---- #
st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
st.write('---')

# ---- APP DESCRIPTION ---- #
st.markdown("""
            This app allows you to train machine learning models for regression and classification tasks.
            Furthermore, it allows you to visualize its performance and compute feature importance using 
            various techniques such as SHAP, Recursive Feature Elimination (RFE), and Boruta algorithm. 
            Using the most important features computed using these techniques, you can train a final 
            machine learning model that can be used for forecasting future trends in your data. This
            app also allows you to train a standalone univariate time series forecasting model using
            LSTM networks.
""")

st.write('---')

# ---- DATA UPLOAD ---- #
with st.container():
    st.write("Upload your data here and specify the independent variables, target variable, and time variable below:")
    data = st.file_uploader("Input Data", type="csv")

    st.session_state.original_df = data_file_loader(data, temp_dir=TEMP_DIR)

    if st.session_state.original_df is not None:
        st.session_state.all_col_names = st.session_state.original_df.columns
    
    if st.session_state.all_col_names is not None:
        st.session_state.time_col = st.selectbox("Column Indicating Time", options=st.session_state.all_col_names)
        st.session_state.independent_feats = st.multiselect("Independent Features", options=st.session_state.all_col_names)
        st.session_state.target_var = st.selectbox("Target Variable", options=st.session_state.all_col_names)

    if ((st.session_state.time_col is not None) and
        (st.session_state.independent_feats != []) and
        (st.session_state.target_var is not None)):
        st.session_state.dataframe = create_ordered_dataframe(st.session_state.original_df, st.session_state.time_col, 
                                                              st.session_state.independent_feats, st.session_state.target_var)
        
        if st.button("Submit Features"):
            st.session_state.processed_df = st.session_state.dataframe.copy(deep=True)
            st.success("Features Submitted!")