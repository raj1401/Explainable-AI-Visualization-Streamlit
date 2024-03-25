import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
import numpy as np
from model_creation_and_training import (create_and_search_tree_classifier, create_and_search_tree_regressor,
                                         create_and_search_logistic_regression, create_and_search_linear_regression,
                                         create_and_Search_KNN_classifer, create_and_Search_KNN_regressor,
                                         create_and_Search_SVM_classifer, create_and_Search_SVM_regressor)
from model_creation_and_training import (train_final_classifier, train_final_regressor, train_final_logistic_regression,
                                         train_final_linear_regression, train_final_KNN_classifier, train_final_KNN_regressor,
                                         train_final_SVM_classifier, train_final_SVM_regressor)

# ---- FUNCTIONS ---- #
def reset_params():
    st.session_state.search_results = {}
    st.session_state.final_params = {}
    st.session_state.dataframe = None
    st.session_state.feats_selected_df = None
    st.session_state.final_model = None
    st.session_state.model_on_selected_feats = None
    st.session_state.TRAIN_TEST_RANDOM_STATE = None
    st.session_state.TEST_FRACTION = None


def on_click_search_params():
    if st.session_state.model_type == "Random Forest Classifier":
        # Change this later
        param_distributions = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_child_samples':[3, 5, 7, 9]
        #'min_data_in_leaf': [3, 5]
        }

        with st.spinner("Searching for Optimum Parameters"):
            params_, model_ = create_and_search_tree_classifier(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_
    
    elif st.session_state.model_type == "Random Forest Regressor":
        # Change this later
        param_distributions = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_child_samples':[3, 5, 7, 9]
        #'min_data_in_leaf': [3, 5]
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_search_tree_regressor(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_
    
    elif st.session_state.model_type == "Logistic Regression":
        param_distributions = {
            'logistic_regression__C': np.arange(0, 10, 0.2),
            'logistic_regression__l1_ratio': np.arange(0, 1, 0.1)
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_search_logistic_regression(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_
    
    elif st.session_state.model_type == "Linear Regression":
        param_distributions = {
            'linear_regression__alpha': np.arange(0, 10, 0.2),
            'linear_regression__l1_ratio': np.arange(0, 1, 0.1)
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_search_linear_regression(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_

    elif st.session_state.model_type == "KNN Classifier":
        param_distributions = {
            'knn_classifier__n_neighbors': np.arange(start=2, stop=16, step=2),
            'knn_classifier__leaf_size': np.arange(start=15, stop=60, step=15),
            'knn_classifier__p': np.arange(start=1, stop=2, step=0.1)
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_KNN_classifer(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_
    
    elif st.session_state.model_type == "KNN Regressor":
        param_distributions = {
            'knn_regressor__n_neighbors': np.arange(start=2, stop=16, step=2),
            'knn_regressor__leaf_size': np.arange(start=15, stop=60, step=15),
            'knn_regressor__p': np.arange(start=1, stop=2, step=0.1)
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_KNN_regressor(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_
    
    elif st.session_state.model_type == "SVM Classifier":
        param_distributions = {
            'svm_classifier__C': np.arange(start=0, stop=10, step=0.2)
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_SVM_classifer(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_
    
    elif st.session_state.model_type == "SVM Regressor":
        param_distributions = {
            'svm_regressor__C': np.arange(start=0, stop=10, step=0.2)
        }

        with st.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_SVM_regressor(df=st.session_state.processed_df, param_distributions=param_distributions)
        st.session_state.search_results["params"] = params_
        st.session_state.search_results["model"] = model_


def train_final_model():
    if st.session_state.model_type == "Random Forest Classifier":
        final_model, rand_state, test_fraction = train_final_classifier(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "Random Forest Regressor":
        final_model, rand_state, test_fraction = train_final_regressor(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "Logistic Regression":
        final_model, rand_state, test_fraction = train_final_logistic_regression(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "Linear Regression":
        final_model, rand_state, test_fraction = train_final_linear_regression(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "KNN Classifier":
        final_model, rand_state, test_fraction = train_final_KNN_classifier(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "KNN Regressor":
        final_model, rand_state, test_fraction = train_final_KNN_regressor(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "SVM Classifier":
        final_model, rand_state, test_fraction = train_final_SVM_classifier(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction
    
    elif st.session_state.model_type == "SVM Regressor":
        final_model, rand_state, test_fraction = train_final_SVM_regressor(df=st.session_state.processed_df, param_distributions=st.session_state.final_params)
        st.session_state.final_model = final_model
        st.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        st.session_state.TEST_FRACTION = test_fraction


def show_model_training_page():
    # # ---- PAGE CONFIG ---- #
    # st.set_page_config(page_title="Explainable AI", layout='wide')

    # # ---- TITLE ---- #
    # st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
    # st.write('---')


    # ---- MODEL SELECTION AND TRAINING ---- #
    st.subheader("Model Selection")
    if st.session_state.processed_df is not None:
        st.write("Now you must select the model you want to use for training, based on the task at hand.")
        st.write("Models Available:")
        model_table = {"Classification Models":st.session_state.classification_models, "Regression Models":st.session_state.regression_models}
        st.table(model_table)
        st.session_state.model_type = st.selectbox(label="Model Type", options=st.session_state.all_models, on_change=reset_params)
        model_submit_button = st.button("Submit Model Type", use_container_width=True, on_click=on_click_search_params)

    # ---- FINAL MODEL TRAINING PARAMETERS ---- #
    with st.container():
        if "params" in st.session_state.search_results:
            st.markdown("<h3 style='text-align: center;'> Hyperparmeter Tuning </h3>", unsafe_allow_html=True)
            st.write("""
                    The following parameters were found optimum using Random Search
                    using five-fold cross validation. You can keep these parameters 
                    or set them yourself. 
            """)
            left_param_col, right_param_col = st.columns((1,1))
            param_dict = st.session_state.search_results["params"]
            num_params = len(param_dict)
            param_keys = list(param_dict.keys())

            for i in range(num_params//2+1):
                with left_param_col:
                    st.session_state.final_params[param_keys[i]] = st.number_input(param_keys[i], value=param_dict[param_keys[i]])
            for i in range(num_params//2+1,num_params):
                with right_param_col:
                    st.session_state.final_params[param_keys[i]] = st.number_input(param_keys[i], value=param_dict[param_keys[i]])

    with st.container():
        if len(st.session_state.final_params) != 0:
            st.write(""" Click on the button below to train the model on these hyperparameters. You will then be able to
                    proceed with feature selection or skip directly to visualization of model performance and prediction.
            """)
            final_model_train_button = st.button("Train Final Model", use_container_width=True, on_click=train_final_model)
        if st.session_state.final_model is not None:
            st.success("Final Model Trained!")