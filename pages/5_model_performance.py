import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
from model_evaluation import (plot_roc_auc, plot_precision_recall, plot_confusion_matrix,
                              get_classification_time_series_predictions, plot_prediction_error,
                              get_regression_metrics, get_regression_time_series_predictions)


# ---- FUNCTIONS ---- #
def plot_classifier_performance_graphs(left_col, middle_col, right_col, df, model):
    # ROC-AUC Curve
    with left_col:
        left_col.markdown("<h3 style='text-align: center;'> ROC-AUC </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_roc_auc(df=df, _model=model, 
                                    random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, 
                                    test_fraction=st.session_state.TEST_FRACTION)
        if fig is None:
            left_col.error(err_msg)
        else:
            left_col.write(fig)
    
    # Precision-Recall Curve
    with middle_col:
        middle_col.markdown("<h3 style='text-align: center;'> Precision-Recall </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_precision_recall(df=df, _model=model, 
                                                random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, 
                                                test_fraction=st.session_state.TEST_FRACTION)
        if fig is None:
            middle_col.error(err_msg)
        else:
            middle_col.write(fig)
    
    # Confusion Matrix
    with right_col:
        right_col.markdown("<h3 style='text-align: center;'> Confusion Matrix </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_confusion_matrix(df=df, _model=model, 
                                                random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, 
                                                test_fraction=st.session_state.TEST_FRACTION)
        if fig is None:
            right_col.error(err_msg)
        else:
            right_col.write(fig)


def plot_classification_time_series_predictions(df, model, col):
    fig, err_msg = get_classification_time_series_predictions(df=df, _model=model,
                                               random_state=st.session_state.TRAIN_TEST_RANDOM_STATE,
                                               test_fraction=st.session_state.TEST_FRACTION)
    if fig is None:
        col.write(err_msg)
    else:
        col.write(fig)


def plot_regressor_performance_graphs(left_col, right_col, df, model):
    # Prediction Error
    with left_col:
        left_col.markdown("<h3 style='text-align: center;'> Prediction Error Graph (On Testing Data) </h3>", unsafe_allow_html=True)
        fig, err_msg = plot_prediction_error(df=df, _model=model, 
                                             random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, 
                                             test_fraction=st.session_state.TEST_FRACTION)
        if fig is None:
            left_col.error(err_msg)
        else:
            left_col.write(fig)
    
    # RMSE On Train and Test Set
    with right_col:
        right_col.markdown("<h3 style='text-align: center;'> Regression Metrics </h3>", unsafe_allow_html=True)
        exp_var_score, mse, r2_scr, err_msg = get_regression_metrics(df=df, _model=model, 
                                             random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, 
                                             test_fraction=st.session_state.TEST_FRACTION)
        if err_msg is None:
            right_col.markdown(f"<h5 style='text-align: center;'> <i> Explained Variance Score </i> = {exp_var_score} </h5>", unsafe_allow_html=True)
            right_col.markdown(f"<h5 style='text-align: center;'> <i> Mean Squared Error </i> = {mse} </h5>", unsafe_allow_html=True)
            right_col.markdown(f"<h5 style='text-align: center;'> <i> R2 Score </i> = {r2_scr} </h5>", unsafe_allow_html=True)
        else:
            right_col.write(err_msg)


def plot_regression_time_series_predictions(df, model, col):
    fig, err_msg = get_regression_time_series_predictions(df=df, _model=model,
                                               random_state=st.session_state.TRAIN_TEST_RANDOM_STATE,
                                               test_fraction=st.session_state.TEST_FRACTION)
    if fig is None:
        col.write(err_msg)
    else:
        col.write(fig)


# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="Explainable AI", layout='wide')

# ---- TITLE ---- #
st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
st.write('---')

# ---- MODEL PERFORMANCE ---- #
st.subheader("Model Performance")
with st.container():
    if st.session_state.model_on_selected_feats is not None:
        # Model Performance
        if st.session_state.perform_feature_selection == "Yes":
            st.markdown("<h2 style='text-align: center;'> Model Performance on Selected Features </h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center;'> Original Model Performance </h2>", unsafe_allow_html=True)
        if st.session_state.model_type in st.session_state.classification_models:
            left_graph_col, middle_graph_col, right_graph_col = st.columns(3)
            plot_classifier_performance_graphs(left_graph_col, middle_graph_col, right_graph_col, st.session_state.feats_selected_df, st.session_state.model_on_selected_feats)
            # Time-Series Prediction
            st.markdown("<h3 style='text-align: center;'> Time-Series Prediction </h3>", unsafe_allow_html=True)
            _, pred_graph_col, _ = st.columns((1,4,1))
            plot_classification_time_series_predictions(st.session_state.feats_selected_df, st.session_state.model_on_selected_feats, pred_graph_col)
        elif st.session_state.model_type in st.session_state.regression_models:
            left_graph_col, right_graph_col = st.columns((1,1))
            plot_regressor_performance_graphs(left_graph_col, right_graph_col, st.session_state.feats_selected_df, st.session_state.model_on_selected_feats)
            # Time-Series Prediction
            st.markdown("<h3 style='text-align: center;'> Time-Series Prediction </h3>", unsafe_allow_html=True)
            _, pred_graph_col, _ = st.columns((1,4,1))
            plot_regression_time_series_predictions(st.session_state.feats_selected_df, st.session_state.model_on_selected_feats, pred_graph_col)