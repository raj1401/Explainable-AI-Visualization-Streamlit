import streamlit as sl
import shap
import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing import get_preprocessing_needs_table, plot_data
from data_preprocessing import fill_null_values, fix_inconsistent_types, remove_duplicates, remove_outliers, scale_features, fix_class_imbalance, interpolate_data

from model_creation_and_training import create_and_search_tree_classifier, train_final_classifier
from model_creation_and_training import create_and_search_tree_regressor, train_final_regressor

from model_creation_and_training import create_and_search_logistic_regression, train_final_logistic_regression
from model_creation_and_training import create_and_search_linear_regression, train_final_linear_regression

from model_creation_and_training import create_and_Search_KNN_classifer, train_final_KNN_classifier
from model_creation_and_training import create_and_Search_KNN_regressor, train_final_KNN_regressor

from model_creation_and_training import create_and_Search_SVM_classifer, train_final_SVM_classifier
from model_creation_and_training import create_and_Search_SVM_regressor, train_final_SVM_regressor

from model_creation_and_training import get_lead_lag_correlations, shift_dataframe

from model_evaluation import plot_precision_recall, plot_roc_auc, plot_confusion_matrix
from model_evaluation import plot_shap_bar, plot_shap_beeswarm, plot_shap_heatmap
from model_evaluation import get_rfe_features, get_boruta_features
from model_evaluation import get_regression_time_series_predictions, get_classification_time_series_predictions
from model_evaluation import plot_prediction_error, get_regression_metrics

from model_forecasting import forecast_from_classifier, forecast_from_regressor, lstm_regression_forecasting

from helper_functions import delete_folder_contents, data_file_loader, create_ordered_dataframe, multi_time_series_loader


# ---- GLOBAL VARIABLES ---- #
TEMP_DIR = "temp_data"

if 'classification_models' not in sl.session_state:
    sl.session_state.classification_models = ["Random Forest Classifier", "Logistic Regression", "KNN Classifier", "SVM Classifier"]

if 'regression_models' not in sl.session_state:
    sl.session_state.regression_models = ["Random Forest Regressor", "Linear Regression", "KNN Regressor", "SVM Regressor"]

if 'all_models' not in sl.session_state:
    sl.session_state.all_models = sl.session_state.classification_models + sl.session_state.regression_models

if 'available_feat_select_algos' not in sl.session_state:
    sl.session_state.available_feat_select_algos = ["SHAP", "Recursive Feature Elimination", "Boruta"]

if 'original_df' not in sl.session_state:
    sl.session_state.original_df = None

if 'all_col_names' not in sl.session_state:
    sl.session_state.all_col_names = None

if 'time_col' not in sl.session_state:
    sl.session_state.time_col = None

if 'independent_feats' not in sl.session_state:
    sl.session_state.independent_feats = None

if 'target_var' not in sl.session_state:
    sl.session_state.target_var = None

if 'search_results' not in sl.session_state:
    sl.session_state.search_results = {}

if 'final_params' not in sl.session_state:
    sl.session_state.final_params = {}

if 'dataframe' not in sl.session_state:
    sl.session_state.dataframe = None

if 'processed_df' not in sl.session_state:
    sl.session_state.processed_df = None

if 'feats_selected_df' not in sl.session_state:
    sl.session_state.feats_selected_df = None

if 'final_model' not in sl.session_state:
    sl.session_state.final_model = None

if 'model_on_selected_feats' not in sl.session_state:
    sl.session_state.model_on_selected_feats = None

if 'trigger_training_on_new_feats' not in sl.session_state:
    sl.session_state.trigger_training_on_new_feats = False

if 'TRAIN_TEST_RANDOM_STATE' not in sl.session_state:
    sl.session_state.TRAIN_TEST_RANDOM_STATE = None

if 'TEST_FRACTION' not in sl.session_state:
    sl.session_state.TEST_FRACTION = None

if 'LSTM_Model' not in sl.session_state:
    sl.session_state.LSTM_Model = None


# ---- HELPER FUNCTIONS ---- #
def on_click_search_params():
    if model_type == "Random Forest Classifier":
        # Change this later
        param_distributions = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_child_samples':[3, 5, 7, 9]
        #'min_data_in_leaf': [3, 5]
        }

        with sl.spinner("Searching for Optimum Parameters"):
            params_, model_ = create_and_search_tree_classifier(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_
    
    elif model_type == "Random Forest Regressor":
        # Change this later
        param_distributions = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_child_samples':[3, 5, 7, 9]
        #'min_data_in_leaf': [3, 5]
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_search_tree_regressor(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_
    
    elif model_type == "Logistic Regression":
        param_distributions = {
            'logistic_regression__C': np.arange(0, 10, 0.2),
            'logistic_regression__l1_ratio': np.arange(0, 1, 0.1)
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_search_logistic_regression(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_
    
    elif model_type == "Linear Regression":
        param_distributions = {
            'linear_regression__alpha': np.arange(0, 10, 0.2),
            'linear_regression__l1_ratio': np.arange(0, 1, 0.1)
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_search_linear_regression(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_

    elif model_type == "KNN Classifier":
        param_distributions = {
            'knn_classifier__n_neighbors': np.arange(start=2, stop=16, step=2),
            'knn_classifier__leaf_size': np.arange(start=15, stop=60, step=15),
            'knn_classifier__p': np.arange(start=1, stop=2, step=0.1)
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_KNN_classifer(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_
    
    elif model_type == "KNN Regressor":
        param_distributions = {
            'knn_regressor__n_neighbors': np.arange(start=2, stop=16, step=2),
            'knn_regressor__leaf_size': np.arange(start=15, stop=60, step=15),
            'knn_regressor__p': np.arange(start=1, stop=2, step=0.1)
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_KNN_regressor(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_
    
    elif model_type == "SVM Classifier":
        param_distributions = {
            'svm_classifier__C': np.arange(start=0, stop=10, step=0.2)
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_SVM_classifer(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_
    
    elif model_type == "SVM Regressor":
        param_distributions = {
            'svm_regressor__C': np.arange(start=0, stop=10, step=0.2)
        }

        with sl.spinner("Searching for Optimum Parameters:"):
            params_, model_ = create_and_Search_SVM_regressor(df=sl.session_state.processed_df, param_distributions=param_distributions)
        sl.session_state.search_results["params"] = params_
        sl.session_state.search_results["model"] = model_

def reset_params():
    sl.session_state.search_results = {}
    sl.session_state.final_params = {}
    sl.session_state.dataframe = None
    sl.session_state.feats_selected_df = None
    sl.session_state.final_model = None
    sl.session_state.model_on_selected_feats = None
    sl.session_state.TRAIN_TEST_RANDOM_STATE = None
    sl.session_state.TEST_FRACTION = None

def reset_on_change_feat_select():
    sl.session_state.feats_selected_df = None
    sl.session_state.model_on_selected_feats = None


def plot_initial_data(plt_col):
    fig, err_msg = plot_data(sl.session_state.processed_df)
    if fig is None:
        plt_col.error(err_msg)
    else:
        plt_col.write(fig)


def remove_null_values_in_data():
    sl.session_state.processed_df = fill_null_values(sl.session_state.processed_df)


def fix_inconsistent_types_in_data():
    sl.session_state.processed_df = fix_inconsistent_types(sl.session_state.processed_df)


def remove_duplicates_in_data():
    sl.session_state.processed_df = remove_duplicates(sl.session_state.processed_df)


def remove_outliers_in_data():
    sl.session_state.processed_df = remove_outliers(sl.session_state.processed_df)


def normalize_data():
    sl.session_state.processed_df = scale_features(sl.session_state.processed_df)


def handle_class_imbalance():
    sl.session_state.processed_df = fix_class_imbalance(sl.session_state.processed_df)


def make_data_periodic(periodicity):
    sl.session_state.processed_df = interpolate_data(sl.session_state.processed_df, periodicity)


# def write_preprocessing_needs_table():
#     sl.write("The following table shows the preprocessing needs of your data:")
#     preprocessing_table = get_preprocessing_needs_table(sl.session_state.processed_df)
#     sl.table(preprocessing_table.transpose())

#     sl.markdown(preprocessing_table.transpose().style.hide(axis="index").to_html(), unsafe_allow_html=True)

#     cols = sl.columns(5)
#     cols[0].button("Remove Null Values", on_click=remove_null_values_in_data)
#     cols[1].button("Fix Inconsistent Types", on_click=fix_inconsistent_types_in_data)
#     cols[2].button("Remove Duplicates", on_click=remove_duplicates_in_data)
#     cols[3].button("Remove Outliers", on_click=remove_outliers_in_data)
#     cols[4].button("Normalize Data", on_click=normalize_data)


def write_preprocessing_needs_table():
    sl.write("The following table shows the preprocessing needs of your data:")
    preprocessing_table = get_preprocessing_needs_table(sl.session_state.processed_df)
    cols = sl.columns(5)

    cols[0].write("Percentage of Null Values")
    cols[0].write(preprocessing_table.iloc[0,1])
    cols[0].button("Remove Null Values", on_click=remove_null_values_in_data)

    cols[1].write("Inconsistent Types")
    cols[1].write(preprocessing_table.iloc[1,1])
    cols[1].button("Fix Inconsistent Types", on_click=fix_inconsistent_types_in_data)

    cols[2].write("Percentage of Duplicates")
    cols[2].write(preprocessing_table.iloc[2,1])
    cols[2].button("Remove Duplicates", on_click=remove_duplicates_in_data)

    cols[3].write("Percentage of Outliers")
    cols[3].write(preprocessing_table.iloc[3,1])
    cols[3].button("Remove Outliers", on_click=remove_outliers_in_data)

    cols[4].write("Need for Scaling")
    cols[4].write(preprocessing_table.iloc[4,1])
    cols[4].button("Normalize Data", on_click=normalize_data)

    second_row_cols = sl.columns(5)

    second_row_cols[0].write("Data Type")
    second_row_cols[0].write(preprocessing_table.iloc[5,1])
    if preprocessing_table.iloc[5,1] == "Data suitable for Classification but needs to handle class imbalance":
        # second_row_cols[0].button("Fix Imbalance", on_click=handle_class_imbalance)
        second_row_cols[0].button("Fix Imbalance")
    
    second_row_cols[1].write("Periodicity in Data")
    second_row_cols[1].write(preprocessing_table.iloc[6,1])
    periodicity = second_row_cols[1].selectbox("Select Periodicity in Days", options=range(1,30))
    second_row_cols[1].button("Make Data Periodic", on_click=make_data_periodic, args=(f"{periodicity}D",))

    second_row_cols[2].write("One-Hot Encoding")

    second_row_cols[3].write("Label Encoding")

    second_row_cols[4].write("One More Function")


def train_final_model():
    if model_type == "Random Forest Classifier":
        final_model, rand_state, test_fraction = train_final_classifier(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "Random Forest Regressor":
        final_model, rand_state, test_fraction = train_final_regressor(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "Logistic Regression":
        final_model, rand_state, test_fraction = train_final_logistic_regression(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "Linear Regression":
        final_model, rand_state, test_fraction = train_final_linear_regression(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "KNN Classifier":
        final_model, rand_state, test_fraction = train_final_KNN_classifier(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "KNN Regressor":
        final_model, rand_state, test_fraction = train_final_KNN_regressor(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "SVM Classifier":
        final_model, rand_state, test_fraction = train_final_SVM_classifier(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction
    
    elif model_type == "SVM Regressor":
        final_model, rand_state, test_fraction = train_final_SVM_regressor(df=sl.session_state.processed_df, param_distributions=sl.session_state.final_params)
        sl.session_state.final_model = final_model
        sl.session_state.TRAIN_TEST_RANDOM_STATE = rand_state
        sl.session_state.TEST_FRACTION = test_fraction


def trigger_training_on_selected_features():
    sl.session_state.trigger_training_on_new_feats = True


def train_model_on_selected_feats():
    if model_type == "Random Forest Classifier":
        model, _, _ = train_final_classifier(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "Random Forest Regressor":
        model, _, _ = train_final_regressor(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "Logistic Regression":
        model, _, _ = train_final_logistic_regression(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "Linear Regression":
        model, _, _ = train_final_linear_regression(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "KNN Classifier":
        model, _, _ = train_final_KNN_classifier(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "KNN Regresor":
        model, _, _ = train_final_KNN_regressor(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "SVM Classifier":
        model, _, _ = train_final_SVM_classifier(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model
    elif model_type == "SVM Regressor":
        model, _, _ = train_final_SVM_regressor(df=sl.session_state.feats_selected_df, param_distributions=sl.session_state.final_params)
        sl.session_state.model_on_selected_feats = model


def train_shifted_model(train_df):
    if model_type == "Random Forest Classifier":
        shifted_model, _, _ = train_final_classifier(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "Random Forest Regressor":
        shifted_model, _, _ = train_final_regressor(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "Logistic Regression":
        shifted_model, _, _ = train_final_logistic_regression(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "Linear Regression":
        shifted_model, _, _ = train_final_linear_regression(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "KNN Classifier":
        shifted_model, _, _ = train_final_KNN_classifier(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "KNN Regressor":
        shifted_model, _, _ = train_final_KNN_regressor(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "SVM Classifier":
        shifted_model, _, _ = train_final_SVM_classifier(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model
    elif model_type == "SVM Regressor":
        shifted_model, _, _ = train_final_SVM_regressor(df=train_df, param_distributions=sl.session_state.final_params)
        return shifted_model


def plot_classifier_performance_graphs(left_col, middle_col, right_col, df, model):
    # ROC-AUC Curve
    with left_col:
        left_col.markdown("<h3 style='text-align: center;'> ROC-AUC </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_roc_auc(df=df, _model=model, 
                                    random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, 
                                    test_fraction=sl.session_state.TEST_FRACTION)
        if fig is None:
            left_col.error(err_msg)
        else:
            left_col.write(fig)
    
    # Precision-Recall Curve
    with middle_col:
        middle_col.markdown("<h3 style='text-align: center;'> Precision-Recall </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_precision_recall(df=df, _model=model, 
                                                random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, 
                                                test_fraction=sl.session_state.TEST_FRACTION)
        if fig is None:
            middle_col.error(err_msg)
        else:
            middle_col.write(fig)
    
    # Confusion Matrix
    with right_col:
        right_col.markdown("<h3 style='text-align: center;'> Confusion Matrix </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_confusion_matrix(df=df, _model=model, 
                                                random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, 
                                                test_fraction=sl.session_state.TEST_FRACTION)
        if fig is None:
            right_col.error(err_msg)
        else:
            right_col.write(fig)


def plot_regressor_performance_graphs(left_col, right_col, df, model):
    # Prediction Error
    with left_col:
        left_col.markdown("<h3 style='text-align: center;'> Prediction Error Graph (On Testing Data) </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_prediction_error(df=df, _model=model, 
                                             random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, 
                                             test_fraction=sl.session_state.TEST_FRACTION)
        if fig is None:
            left_col.error(err_msg)
        else:
            left_col.write(fig)
    
    # RMSE On Train and Test Set
    with right_col:
        right_col.markdown("<h3 style='text-align: center;'> Regression Metrics </h3>", unsafe_allow_html=True)
        exp_var_score, mse, r2_scr, err_msg = get_regression_metrics(df=df, _model=model, 
                                             random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, 
                                             test_fraction=sl.session_state.TEST_FRACTION)
        if err_msg is None:
            right_col.markdown(f"<h5 style='text-align: center;'> <i> Explained Variance Score </i> = {exp_var_score} </h5>", unsafe_allow_html=True)
            right_col.markdown(f"<h5 style='text-align: center;'> <i> Mean Squared Error </i> = {mse} </h5>", unsafe_allow_html=True)
            right_col.markdown(f"<h5 style='text-align: center;'> <i> R2 Score </i> = {r2_scr} </h5>", unsafe_allow_html=True)
        else:
            right_col.write(err_msg)


def plot_shap_graphs(left_col, middle_col, right_col, df, model):
    if model_type in sl.session_state.classification_models:
        X_train, X_test, _, _ = train_test_split(df.iloc[:,1:-1], df.iloc[:,-1], test_size=sl.session_state.TEST_FRACTION, 
                                                 shuffle=True, stratify=df.iloc[:,-1], random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE)
    else:
        X_train, X_test, _, _ = train_test_split(df.iloc[:,1:-1], df.iloc[:,-1], test_size=sl.session_state.TEST_FRACTION, 
                                                 shuffle=False, random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE)
    
    if model_type in ["Random Forest Classifier", "Random Forest Regressor"]:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
    elif model_type == "Logistic Regression":
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.Explainer(model.named_steps['logistic_regression'], back_dist)
        shap_values = explainer(X_test)
    elif model_type == "Linear Regression":
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.Explainer(model.named_steps['linear_regression'], back_dist)
        shap_values = explainer(X_test)
    elif model_type == "KNN Classifier":
        # back_dist = X_train.median().values.reshape((1, X_train.shape[1]))
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.KernelExplainer(model.named_steps['knn_classifier'].predict, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)
    elif model_type == "KNN Regressor":
        # back_dist = X_train.median().values.reshape((1, X_train.shape[1]))
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.KernelExplainer(model.named_steps['knn_regressor'].predict, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)
    elif model_type == "SVM Classifier":
        back_dist = shap.utils.sample(X_train, int(0.05*len(X_train.iloc[:,0])))
        # explainer = shap.Explainer(model.named_steps['svm_classifier'], back_dist)
        explainer = shap.KernelExplainer(model.named_steps['svm_classifier'].predict_proba, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)
    elif model_type == "SVM Regressor":
        back_dist = shap.utils.sample(X_train, int(0.05*len(X_train.iloc[:,0])))
        # explainer = shap.Explainer(model.named_steps['svm_classifier'], back_dist)
        explainer = shap.KernelExplainer(model.named_steps['svm_regressor'].predict, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)

    print(shap_values)
    # Bar Plot
    with left_col:
        left_col.markdown("<h4 style='text-align: center;'> Bar </h4>", unsafe_allow_html=True)
        fig, err_msg = plot_shap_bar(shap_values, max_display=len(X_test.columns))
        if fig is None:
            left_col.error(err_msg)
        else:
            left_col.write(fig)
    
    # Beeswarm Plot
    with middle_col:
        middle_col.markdown("<h4 style='text-align: center;'> Beeswarm </h4>", unsafe_allow_html=True)
        fig, err_msg = plot_shap_beeswarm(shap_values, max_display=len(X_test.columns))
        if fig is None:
            middle_col.error(err_msg)
        else:
            middle_col.write(fig)

    # Heatmap
    with right_col:
        right_col.markdown("<h4 style='text-align: center;'> Heatmap </h4>", unsafe_allow_html=True)
        fig, err_msg = plot_shap_heatmap(shap_values, max_display=len(X_test.columns))
        if fig is None:
            right_col.error(err_msg)
        else:
            right_col.write(fig)


def plot_rfe_features(col, df, model):
    with col:
        fig, err_msg = get_rfe_features(df, model, model_type, random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE,
                                        test_fraction=sl.session_state.TEST_FRACTION)
        if fig is None:
            col.error(err_msg)
        else:
            col.write(fig)


def plot_boruta_features(col, df):
    with col:
        p_val = sl.slider("Significance (p-value)", min_value=0.05, max_value=0.95, step=0.05)
        fig, err_msg = get_boruta_features(df, model_type=model_type, random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE,
                                           test_fraction=sl.session_state.TEST_FRACTION,
                                           param_distributions=sl.session_state.final_params,
                                           p_val=p_val)
        if fig is None:
            col.error(err_msg)
        else:
            col.write(fig)



def plot_regression_time_series_predictions(df, model, col):
    fig, err_msg = get_regression_time_series_predictions(df=df, _model=model,
                                               random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE,
                                               test_fraction=sl.session_state.TEST_FRACTION)
    if fig is None:
        col.write(err_msg)
    else:
        col.write(fig)


def plot_classification_time_series_predictions(df, model, col):
    fig, err_msg = get_classification_time_series_predictions(df=df, _model=model,
                                               random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE,
                                               test_fraction=sl.session_state.TEST_FRACTION)
    if fig is None:
        col.write(err_msg)
    else:
        col.write(fig)


def plot_forecasts(new_data, left_col, right_col):
    if model_type in sl.session_state.classification_models:
        fig, df, err_msg = forecast_from_classifier(_model=sl.session_state.model_on_selected_feats, data_file=new_data)
        if err_msg is not None:
            sl.write(err_msg)
        else:
            left_col.write(fig)
            csv_file = df.to_csv(index=False).encode('utf-8')
            right_col.write("You can download the augmented dataset where the last column contains the predicted values:")
            with right_col:
                sl.download_button(label="Download Augmented Dataset", data=csv_file, file_name="augmented_dataset.csv", mime='text/csv')

    elif model_type in sl.session_state.regression_models:
        fig, df, err_msg = forecast_from_regressor(_model=sl.session_state.model_on_selected_feats, data_file=new_data)
        if err_msg is not None:
            sl.write(err_msg)
        else:
            left_col.write(fig)
            csv_file = df.to_csv(index=False).encode('utf-8')
            right_col.write("You can download the augmented dataset where the last column contains the predicted values:")
            with right_col:
                sl.download_button(label="Download Augmented Dataset", data=csv_file, file_name="augmented_dataset.csv", mime='text/csv')


def plot_lstm_forecasts_single(input_data_file, future_steps, plot_col, batch_size, lookback, hidden_size, num_layers):
    input_df = data_file_loader(input_data_file, temp_dir=TEMP_DIR)
    fig, err_msg, trained_lstm = lstm_regression_forecasting(df=input_df, test_fraction=sl.session_state.TEST_FRACTION,
                                                            random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, from_df=True,
                                                            future_steps=future_steps, lstm_model=sl.session_state.LSTM_Model,
                                                            batch_size=batch_size, lookback=lookback, hidden_size=hidden_size,
                                                            num_layers=num_layers)
    sl.session_state.LSTM_Model = trained_lstm
    if fig is None:
        plot_col.write(err_msg)
    else:
        plot_col.write(fig)


def plot_lstm_forecasts_multiple(input_data_file, future_steps, plot_col, batch_size, hidden_size, num_layers):
    y_list = multi_time_series_loader(input_data_file, temp_dir=TEMP_DIR)
    fig, err_msg, trained_lstm = lstm_regression_forecasting(df=None, y_list=y_list, test_fraction=sl.session_state.TEST_FRACTION,
                                                            random_state=sl.session_state.TRAIN_TEST_RANDOM_STATE, from_df=False,
                                                            future_steps=future_steps, lstm_model=sl.session_state.LSTM_Model,
                                                            batch_size=batch_size, hidden_size=hidden_size,
                                                            num_layers=num_layers)
    sl.session_state.LSTM_Model = trained_lstm
    if fig is None:
        plot_col.write(err_msg)
    else:
        plot_col.write(fig)


def reset_lstm_model():
    sl.session_state.LSTM_Model = None


# ---- PAGE CONFIG ---- #
sl.set_page_config(page_title="Explainable AI", layout='wide')

# ---- HEADER SECTION ---- #
sl.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
# sl.markdown("<h3 style='text-align: center;'> Explainable AI: Modeling and Visualization </h1>", unsafe_allow_html=True)
# sl.markdown("<h5 style='text-align: center;'> Made By Raj Mehta </h5>", unsafe_allow_html=True)

sl.write('---')

sl.markdown("""
            This app allows you to train machine learning models for regression and classification tasks.
            Furthermore, it allows you to visualize its performance and compute feature importance using 
            various techniques such as SHAP, Recursive Feature Elimination (RFE), and Boruta algorithm. 
            Using the most important features computed using these techniques, you can train a final 
            machine learning model that can be used for forecasting future trends in your data. This
            app also allows you to train a standalone univariate time series forecasting model using
            LSTM networks.
""")

sl.write('---')

# ---- INPUT PARAMETERS ---- #
sl.subheader("Input Data")

# with sl.container():
#     data_col, model_col = sl.columns((3,2))
#     with data_col:
#         data = sl.file_uploader("Input Data", type="csv")
    
#     with model_col:
#         model_type = sl.selectbox(label="Model Type", options=["Random Forest Classifier", "Random Forest Regressor"], key="model_type", on_change=reset_params)
#         model_submit_button = sl.button("Submit Model Type", use_container_width=True, on_click=on_click_search_params)


with sl.container():
    sl.write("Upload your data here and specify the independent variables, target variable, and time variable below:")
    data = sl.file_uploader("Input Data", type="csv")

    sl.session_state.original_df = data_file_loader(data, temp_dir=TEMP_DIR)

    if sl.session_state.original_df is not None:
        sl.session_state.all_col_names = sl.session_state.original_df.columns
    
    if sl.session_state.all_col_names is not None:
        sl.session_state.time_col = sl.selectbox("Column Indicating Time", options=sl.session_state.all_col_names)
        sl.session_state.independent_feats = sl.multiselect("Independent Features", options=sl.session_state.all_col_names)
        sl.session_state.target_var = sl.selectbox("Target Variable", options=sl.session_state.all_col_names)

    if ((sl.session_state.time_col is not None) and
        (sl.session_state.independent_feats != []) and
        (sl.session_state.target_var is not None)):
        sl.session_state.dataframe = create_ordered_dataframe(sl.session_state.original_df, sl.session_state.time_col, 
                                                              sl.session_state.independent_feats, sl.session_state.target_var)
        
        if sl.button("Submit Features"):
            sl.session_state.processed_df = sl.session_state.dataframe.copy(deep=True)
            sl.success("Features Submitted!")
        sl.subheader("Data Preprocessing")
        # _, preprocess_col, _ = sl.columns((1,4,1))
        # preprocess_button = sl.button("Check Preprocessing Needs", use_container_width=True)
        if sl.session_state.processed_df is not None:
            _, plt_col, _ = sl.columns((1,4,1))
            plot_initial_data(plt_col)
            write_preprocessing_needs_table()
        # if sl.button("Check Preprocessing Needs", use_container_width=True):
        #     _, plt_col, _ = sl.columns((1,4,1))
        #     plot_initial_data(plt_col)
        #     write_preprocessing_needs_table()

        # with preprocess_col:
        #     sl.button("Check Preprocessing Needs", use_container_width=True, on_click=write_preprocessing_needs_table)
        
        sl.subheader("Model Selection")
        sl.write("Now you must select the model you want to use for training, based on the task at hand.")
        sl.write("Models Available:")
        model_table = {"Classification Models":sl.session_state.classification_models, "Regression Models":sl.session_state.regression_models}
        sl.table(model_table)
        # sl.write(f"Classification Models available: {sl.session_state.classification_models}")
        # sl.write(f"Regression Models available: {sl.session_state.regression_models}")

        model_type = sl.selectbox(label="Model Type", options=sl.session_state.all_models, on_change=reset_params)
        model_submit_button = sl.button("Submit Model Type", use_container_width=True, on_click=on_click_search_params)





# ---- FINAL MODEL TRAINING PARAMETERS ---- #
with sl.container():
    if "params" in sl.session_state.search_results:
        sl.markdown("<h3 style='text-align: center;'> Hyperparmeter Tuning </h3>", unsafe_allow_html=True)
        sl.write("""
                The following parameters were found optimum using Random Search
                using five-fold cross validation. You can keep these parameters 
                or set them yourself. 
        """)
        left_param_col, right_param_col = sl.columns((1,1))
        param_dict = sl.session_state.search_results["params"]
        num_params = len(param_dict)
        param_keys = list(param_dict.keys())

        for i in range(num_params//2+1):
            with left_param_col:
                sl.session_state.final_params[param_keys[i]] = sl.number_input(param_keys[i], value=param_dict[param_keys[i]])
        for i in range(num_params//2+1,num_params):
            with right_param_col:
                sl.session_state.final_params[param_keys[i]] = sl.number_input(param_keys[i], value=param_dict[param_keys[i]])

with sl.container():
    if len(sl.session_state.final_params) != 0:
        sl.write(""" Click on the button below to train the model on these hyperparameters. You will then be able to
                 proceed with feature selection or skip directly to visualization of model performance and prediction.
        """)
        final_model_train_button = sl.button("Train Final Model", use_container_width=True, on_click=train_final_model)
    if sl.session_state.final_model is not None:
        sl.success("Final Model Trained!")


# # ---- MODEL EVALUATION AND PERFORMANCE ---- #
# with sl.container():
#     if sl.session_state.final_model is not None:        
#         # # Model Performance
#         # sl.markdown("<h2 style='text-align: center;'> Model Performance </h2>", unsafe_allow_html=True)
#         # if model_type == "Random Forest Classifier":
#         #     left_graph_col, middle_graph_col, right_graph_col = sl.columns(3)
#         #     plot_classifier_performance_graphs(left_graph_col, middle_graph_col, right_graph_col, sl.session_state.dataframe, sl.session_state.final_model)
#         # elif model_type == "Random Forest Regressor":
#         #     left_graph_col, right_graph_col = sl.columns((1,1))
#         #     plot_regressor_performance_graphs(left_graph_col, right_graph_col, sl.session_state.dataframe, sl.session_state.final_model)

#         # SHAP Values
#         sl.markdown("<h2 style='text-align: center;'> SHAP Values </h2>", unsafe_allow_html=True)
#         left_shap_col, middle_shap_col, right_shap_col = sl.columns(3)
#         plot_shap_graphs(left_shap_col, middle_shap_col, right_shap_col, sl.session_state.dataframe, sl.session_state.final_model)


# ---- FEATURE SELECTION ---- #
with sl.container():
    if sl.session_state.final_model is not None:
        sl.markdown("<h4> Do you want to perform feature selection? </h4>", unsafe_allow_html=True)
        perform_feature_selection = sl.selectbox("Perform Feature Selection", options=["Yes", "No"], on_change=reset_on_change_feat_select)

        if perform_feature_selection == "Yes":
            sl.write("Select the feature selection algorithms whose results you want to see:")
            feature_selection_algorithms = sl.multiselect("Feature Selection Algorithms", options=sl.session_state.available_feat_select_algos, on_change=reset_on_change_feat_select)

            if "SHAP" in feature_selection_algorithms:
                # SHAP Values
                sl.markdown("<h3 style='text-align: center;'> SHAP Values </h3>", unsafe_allow_html=True)
                left_shap_col, middle_shap_col, right_shap_col = sl.columns(3)
                plot_shap_graphs(left_shap_col, middle_shap_col, right_shap_col, sl.session_state.processed_df, sl.session_state.final_model)
            
            # IMPLEMENT RFE AND BARUTA LATER
            if "Recursive Feature Elimination" in feature_selection_algorithms:
                # RFE Values
                sl.markdown("<h3 style='text-align: center;'> Recursive Feature Elimination </h3>", unsafe_allow_html=True)
                _, middle_rfe_col, _ = sl.columns(3)
                plot_rfe_features(middle_rfe_col, sl.session_state.processed_df, sl.session_state.final_model)
            
            if "Boruta" in feature_selection_algorithms:
                # Boruta Values
                sl.markdown("<h3 style='text-align: center;'> Boruta Algorithm </h3>", unsafe_allow_html=True)
                _, middle_rfe_col, _ = sl.columns(3)                
                plot_boruta_features(middle_rfe_col, sl.session_state.processed_df)


            if len(feature_selection_algorithms) != 0:
                sl.markdown("""<h4 style='text-align: center;'> Select the most important features that
                    you want to retain in the data based on the above Graphs </h4>""", unsafe_allow_html=True)
                new_features = sl.multiselect("Features", options=sl.session_state.independent_feats, on_change=trigger_training_on_selected_features)
                
                if len(new_features) != 0:
                    sl.session_state.feats_selected_df = create_ordered_dataframe(sl.session_state.processed_df, sl.session_state.time_col, 
                                                                                  new_features, sl.session_state.target_var)
                    # sl.session_state.feats_selected_df = sl.session_state.dataframe.loc[:,new_features].copy(deep=True)
                    # sl.session_state.feats_selected_df[list(sl.session_state.dataframe.columns)[-1]] = sl.session_state.dataframe.iloc[:,-1].copy(deep=True)
                else:
                    sl.session_state.feats_selected_df = None

                if sl.session_state.trigger_training_on_new_feats and len(new_features) != 0:
                    train_model_on_selected_feats()
                    sl.session_state.trigger_training_on_new_feats = False
        else:
            sl.session_state.feats_selected_df = sl.session_state.dataframe
            sl.session_state.model_on_selected_feats = sl.session_state.final_model

        # sl.markdown("<h2 style='text-align: center;'> Feature Selection </h2>", unsafe_allow_html=True)
        # sl.markdown("""<h4 style='text-align: center;'> Select the most important features that
        #             you want to retain in the data based on the above SHAP Graphs </h4>""", unsafe_allow_html=True)

        # new_features = sl.multiselect("Features", options=list(sl.session_state.dataframe.columns[:-1]), on_change=trigger_training_on_selected_features)
        # if len(new_features) != 0:
        #     sl.session_state.feats_selected_df = sl.session_state.dataframe.loc[:,new_features].copy(deep=True)
        #     sl.session_state.feats_selected_df[list(sl.session_state.dataframe.columns)[-1]] = sl.session_state.dataframe.iloc[:,-1].copy(deep=True)
        # else:
        #     sl.session_state.feats_selected_df = None

        # if sl.session_state.trigger_training_on_new_feats and len(new_features) != 0:
        #     train_model_on_selected_feats()
        #     sl.session_state.trigger_training_on_new_feats = False


# ---- MODEL EVALUATION WITH FEATURE SELECTED DATA ---- #
with sl.container():
    if sl.session_state.model_on_selected_feats is not None:
        # Model Performance
        if perform_feature_selection == "Yes":
            sl.markdown("<h2 style='text-align: center;'> Model Performance on Selected Features </h2>", unsafe_allow_html=True)
        else:
            sl.markdown("<h2 style='text-align: center;'> Original Model Performance </h2>", unsafe_allow_html=True)
        if model_type in sl.session_state.classification_models:
            left_graph_col, middle_graph_col, right_graph_col = sl.columns(3)
            plot_classifier_performance_graphs(left_graph_col, middle_graph_col, right_graph_col, sl.session_state.feats_selected_df, sl.session_state.model_on_selected_feats)
            # Time-Series Prediction
            sl.markdown("<h3 style='text-align: center;'> Time-Series Prediction </h3>", unsafe_allow_html=True)
            _, pred_graph_col, _ = sl.columns((1,4,1))
            plot_classification_time_series_predictions(sl.session_state.feats_selected_df, sl.session_state.model_on_selected_feats, pred_graph_col)
        elif model_type in sl.session_state.regression_models:
            left_graph_col, right_graph_col = sl.columns((1,1))
            plot_regressor_performance_graphs(left_graph_col, right_graph_col, sl.session_state.feats_selected_df, sl.session_state.model_on_selected_feats)
            # Time-Series Prediction
            sl.markdown("<h3 style='text-align: center;'> Time-Series Prediction </h3>", unsafe_allow_html=True)
            _, pred_graph_col, _ = sl.columns((1,4,1))
            plot_regression_time_series_predictions(sl.session_state.feats_selected_df, sl.session_state.model_on_selected_feats, pred_graph_col)


        # # SHAP Values
        # sl.markdown("<h3 style='text-align: center;'> SHAP Values of Selected Features </h3>", unsafe_allow_html=True)
        # left_shap_col, middle_shap_col, right_shap_col = sl.columns(3)
        # plot_shap_graphs(left_shap_col, middle_shap_col, right_shap_col, sl.session_state.feats_selected_df, sl.session_state.model_on_selected_feats)


# ---- USING THE MODEL ON NEW DATA ---- #
with sl.container():
    if sl.session_state.model_on_selected_feats is not None:
        sl.markdown("<h2 style='text-align: center;'> Forecasting From The Data </h2>", unsafe_allow_html=True)
        sl.write("""
                 Forecasting from any provided data can be a univariate or a multivariate problem. In the univariate
                 setting, we try to predict the future values of a time-series using only the values it takes in the
                 previous time steps. We accomplish this usually by training a Recurrent Neural Network.
                 On the other hand, in the multivariate setting, we try to predict the future
                 values of a time-series based on other independent variables whose values are/will be known in the future.
        """)
        problem_type = sl.selectbox("Problem Type", options=["Univariate", "Multivariate"], on_change=reset_lstm_model)

        if problem_type == "Univariate":
            sl.write("""We can train an LSTM, a special kind of Recurrent Neural Network (RNN) to predict future values
                     from a periodic time-series data. Do you want to upload a DataFrame (CSV File) to predict future values of the
                     time series it contains in its last column, or do you want to upload a DataFrame (CSV File) that contains
                     multiple time-series data of the same target variable and same periodicity in its columns and train the LSTM on them?
                     The latter is recommended if you have 100s of time-series data.""")
            input_type = sl.selectbox("Input Type", options=["Single DataFrame", "Multiple Data Series"], on_change=reset_lstm_model)
            
            sl.write("How many time steps in the future do you want to make the predictions?")
            f_col, _ = sl.columns((1,3))
            future_steps = f_col.number_input("Future Time Steps", min_value=20, step=1)
            sl.write("Model and Training Hyperparameters")
            col1, col2, col3, col4 = sl.columns(4)
            batch_size = col1.number_input("Batch Size", min_value=4, step=1, on_change=reset_lstm_model)
            hidden_size = col2.number_input("LSTM Hidden State Size", min_value=8, step=1, on_change=reset_lstm_model)
            num_layers = col3.number_input("Number of Hidden LSTM Layers", min_value=1, step=1, on_change=reset_lstm_model)

            if input_type == "Single DataFrame":
                lookback = col4.number_input("Window Size", min_value=1, step=1, on_change=reset_lstm_model)
                single_df_data = sl.file_uploader("Single DataFrame", type="csv")
                if single_df_data is not None:
                    _, plot_col, _ = sl.columns((1,4,1))
                    plot_lstm_forecasts_single(input_data_file=single_df_data, future_steps=future_steps, plot_col=plot_col,
                                               batch_size=batch_size, lookback=lookback, hidden_size=hidden_size, num_layers=num_layers)
            
            if input_type == "Multiple Data Series":
                multi_df_data = sl.file_uploader("Multiple Data Series", type="csv")
                if multi_df_data is not None:
                    _, plot_col, _ = sl.columns((1,4,1))
                    plot_lstm_forecasts_multiple(input_data_file=multi_df_data, future_steps=future_steps, plot_col=plot_col,
                                                 batch_size=batch_size, hidden_size=hidden_size, num_layers=num_layers)

            # if new_data is not None:
            #     _, plot_col, _ = sl.columns((1,4,1))
            #     plot_lstm_forecasts(input_type=input_type, input_data_file=new_data, future_steps=future_steps, plot_col=plot_col)

        if problem_type == "Multivariate":
            # Getting New Data
            sl.write("""You can now use the above trained model to make predictions
                        about the target variable using your own data. Make sure that the data that you upload only has
                        the features that were chosen above to train the final model (the first column should
                        have the time the data points were recorded).""")
            # sl.markdown("""<h5 style='text-align: center;'> You can now use the above trained model to make predictions
            #             about the target variable using your own data. Make sure that the data that you upload only has
            #             the features that were chosen above to train the final model (the first column should
            #             have the time the data points were recorded). </h5>""", unsafe_allow_html= True)

            new_data = sl.file_uploader("New Input Data", type="csv")
            if new_data is not None:
                left_graph_col, right_col = sl.columns((3,1))
                plot_forecasts(new_data, left_graph_col, right_col)



# # Leaving this out for now        
# # ---- LEAD-LAG TESTING AND VISUALIZATION ---- #
# with sl.container():
#     if sl.session_state.final_model is not None:
#         sl.markdown("<h2 style='text-align: center;'> Lead-Lag Correlations </h1>", unsafe_allow_html=True) 
#         sl.write("""
#             We can shift the feature columns by certain amount (3 time steps in our case)
#             and compute Spearman correlation from which we can extract lead and lag
#             correlation values which can help us better understand what feature values
#             precede and succeed other values.
#         """)

#         lead_correlation, lag_correlation = get_lead_lag_correlations(df=sl.session_state.dataframe)
#         df_left_col, df_right_col = sl.columns((1,1))
        
#         with df_left_col:
#             sl.markdown("<h3 style='text-align: center;'> Lead Correlations </h1>", unsafe_allow_html=True) 
#             sl.dataframe(lead_correlation, use_container_width=True)
#         with df_right_col:
#             sl.markdown("<h3 style='text-align: center;'> Lag Correlations </h1>", unsafe_allow_html=True) 
#             sl.dataframe(lag_correlation, use_container_width=True)
        
#         lead_df = shift_dataframe(df=sl.session_state.dataframe, corr=lead_correlation)
#         lag_df = shift_dataframe(df=sl.session_state.dataframe, corr=lag_correlation)

#         with sl.spinner("Training Models on Lead and Lag Data"):
#             lead_model = train_shifted_model(lead_df)
#             lag_model = train_shifted_model(lag_df)
        
#         # Plotting Lead Model Graphs
#         if lead_model is not None:
#             sl.markdown("<h2 style='text-align: center;'> Model Performance on Lead data </h1>", unsafe_allow_html=True)
#             if model_type == "Random Forest Classifier":
#                 left_graph_lead_col, right_graph_lead_col = sl.columns((1,1))

#                 plot_classifier_graphs(left_graph_lead_col, right_graph_lead_col, lead_df, lead_model)
        
#         # Plotting Lag Model Graphs
#         if lag_model is not None:
#             sl.markdown("<h2 style='text-align: center;'> Model Performance on Lag data </h1>", unsafe_allow_html=True)
#             if model_type == "Random Forest Classifier":
#                 left_graph_lag_col, right_graph_lag_col = sl.columns((1,1))

#                 plot_classifier_graphs(left_graph_lag_col, right_graph_lag_col, lag_df, lag_model)


delete_folder_contents(TEMP_DIR)
