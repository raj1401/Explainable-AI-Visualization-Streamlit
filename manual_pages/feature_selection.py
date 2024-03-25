import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from model_evaluation import (plot_shap_bar, plot_shap_beeswarm, plot_shap_heatmap,
                              get_rfe_features, get_boruta_features)
from helper_functions import create_ordered_dataframe
from model_creation_and_training import (train_final_classifier, train_final_regressor, train_final_logistic_regression,
                                         train_final_linear_regression, train_final_KNN_classifier, train_final_KNN_regressor,
                                         train_final_SVM_classifier, train_final_SVM_regressor)


# ---- FUNCTIONS ---- #
def reset_on_change_feat_select():
    st.session_state.feats_selected_df = None
    st.session_state.model_on_selected_feats = None


def plot_shap_graphs(left_col, middle_col, right_col, df, model):
    if st.session_state.model_type in st.session_state.classification_models:
        X_train, X_test, _, _ = train_test_split(df.iloc[:,1:-1], df.iloc[:,-1], test_size=st.session_state.TEST_FRACTION, 
                                                 shuffle=True, stratify=df.iloc[:,-1], random_state=st.session_state.TRAIN_TEST_RANDOM_STATE)
    else:
        X_train, X_test, _, _ = train_test_split(df.iloc[:,1:-1], df.iloc[:,-1], test_size=st.session_state.TEST_FRACTION, 
                                                 shuffle=False, random_state=st.session_state.TRAIN_TEST_RANDOM_STATE)
    
    if st.session_state.model_type in ["Random Forest Classifier", "Random Forest Regressor"]:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
    elif st.session_state.model_type == "Logistic Regression":
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.Explainer(model.named_steps['logistic_regression'], back_dist)
        shap_values = explainer(X_test)
    elif st.session_state.model_type == "Linear Regression":
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.Explainer(model.named_steps['linear_regression'], back_dist)
        shap_values = explainer(X_test)
    elif st.session_state.model_type == "KNN Classifier":
        # back_dist = X_train.median().values.reshape((1, X_train.shape[1]))
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.KernelExplainer(model.named_steps['knn_classifier'].predict, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)
    elif st.session_state.model_type == "KNN Regressor":
        # back_dist = X_train.median().values.reshape((1, X_train.shape[1]))
        back_dist = shap.utils.sample(X_train, int(0.1*len(X_train.iloc[:,0])))
        explainer = shap.KernelExplainer(model.named_steps['knn_regressor'].predict, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)
    elif st.session_state.model_type == "SVM Classifier":
        back_dist = shap.utils.sample(X_train, int(0.05*len(X_train.iloc[:,0])))
        # explainer = shap.Explainer(model.named_steps['svm_classifier'], back_dist)
        explainer = shap.KernelExplainer(model.named_steps['svm_classifier'].predict, back_dist)
        shap_values = explainer.shap_values(X_test)
        base_values = np.array([explainer.expected_value] * len(X_test.iloc[:,0]))
        shap_values = shap.Explanation(values=shap_values, base_values=base_values, data=X_test.to_numpy(), feature_names=X_test.columns)
    elif st.session_state.model_type == "SVM Regressor":
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
        fig, err_msg = get_rfe_features(df, model, st.session_state.model_type, random_state=st.session_state.TRAIN_TEST_RANDOM_STATE,
                                        test_fraction=st.session_state.TEST_FRACTION)
        if fig is None:
            col.error(err_msg)
        else:
            col.write(fig)


def plot_boruta_features(col, df):
    with col:
        p_val = st.slider("Significance (p-value)", min_value=0.05, max_value=0.95, step=0.05)
        fig, err_msg = get_boruta_features(df, model_type=st.session_state.model_type, random_state=st.session_state.TRAIN_TEST_RANDOM_STATE,
                                           test_fraction=st.session_state.TEST_FRACTION,
                                           param_distributions=st.session_state.final_params,
                                           p_val=p_val)
        if fig is None:
            col.error(err_msg)
        else:
            col.write(fig)


def trigger_training_on_selected_features():
    st.session_state.trigger_training_on_new_feats = True

def train_model_on_selected_feats():
    if st.session_state.model_type == "Random Forest Classifier":
        model, _, _ = train_final_classifier(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "Random Forest Regressor":
        model, _, _ = train_final_regressor(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "Logistic Regression":
        model, _, _ = train_final_logistic_regression(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "Linear Regression":
        model, _, _ = train_final_linear_regression(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "KNN Classifier":
        model, _, _ = train_final_KNN_classifier(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "KNN Regresor":
        model, _, _ = train_final_KNN_regressor(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "SVM Classifier":
        model, _, _ = train_final_SVM_classifier(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)
        st.session_state.model_on_selected_feats = model
    elif st.session_state.model_type == "SVM Regressor":
        model, _, _ = train_final_SVM_regressor(df=st.session_state.feats_selected_df, param_distributions=st.session_state.final_params)


def show_feature_selection_page():
    # # ---- PAGE CONFIG ---- #
    # st.set_page_config(page_title="Explainable AI", layout='wide')

    # # ---- TITLE ---- #
    # st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
    # st.write('---')

    # ---- FEATURE SELECTION ---- #
    st.subheader("Feature Selection")
    with st.container():
        if st.session_state.final_model is not None:
            st.markdown("<h4> Do you want to perform feature selection? </h4>", unsafe_allow_html=True)
            st.session_state.perform_feature_selection = st.selectbox("Perform Feature Selection", options=["Yes", "No"], on_change=reset_on_change_feat_select)

            if st.session_state.perform_feature_selection == "Yes":
                st.write("Select the feature selection algorithms whose results you want to see:")
                feature_selection_algorithms = st.multiselect("Feature Selection Algorithms", options=st.session_state.available_feat_select_algos, on_change=reset_on_change_feat_select)

                if "SHAP" in feature_selection_algorithms:
                    # SHAP Values
                    st.markdown("<h3 style='text-align: center;'> SHAP Values </h3>", unsafe_allow_html=True)
                    left_shap_col, middle_shap_col, right_shap_col = st.columns(3)
                    plot_shap_graphs(left_shap_col, middle_shap_col, right_shap_col, st.session_state.processed_df, st.session_state.final_model)

                if "Recursive Feature Elimination" in feature_selection_algorithms:
                    # RFE Values
                    st.markdown("<h3 style='text-align: center;'> Recursive Feature Elimination </h3>", unsafe_allow_html=True)
                    _, middle_rfe_col, _ = st.columns(3)
                    plot_rfe_features(middle_rfe_col, st.session_state.processed_df, st.session_state.final_model)
                
                if "Boruta" in feature_selection_algorithms:
                    # Boruta Values
                    st.markdown("<h3 style='text-align: center;'> Boruta Algorithm </h3>", unsafe_allow_html=True)
                    _, middle_rfe_col, _ = st.columns(3)                
                    plot_boruta_features(middle_rfe_col, st.session_state.processed_df)


                if len(feature_selection_algorithms) != 0:
                    st.markdown("""<h4 style='text-align: center;'> Select the most important features that
                        you want to retain in the data based on the above Graphs </h4>""", unsafe_allow_html=True)
                    new_features = st.multiselect("Features", options=st.session_state.independent_feats, on_change=trigger_training_on_selected_features)
                    
                    if len(new_features) != 0:
                        st.session_state.feats_selected_df = create_ordered_dataframe(st.session_state.processed_df, st.session_state.time_col, 
                                                                                    new_features, st.session_state.target_var)
                        # sl.session_state.feats_selected_df = sl.session_state.dataframe.loc[:,new_features].copy(deep=True)
                        # sl.session_state.feats_selected_df[list(sl.session_state.dataframe.columns)[-1]] = sl.session_state.dataframe.iloc[:,-1].copy(deep=True)
                    else:
                        st.session_state.feats_selected_df = None

                    if st.session_state.trigger_training_on_new_feats and len(new_features) != 0:
                        train_model_on_selected_feats()
                        st.session_state.trigger_training_on_new_feats = False
            else:
                st.session_state.feats_selected_df = st.session_state.dataframe
                st.session_state.model_on_selected_feats = st.session_state.final_model