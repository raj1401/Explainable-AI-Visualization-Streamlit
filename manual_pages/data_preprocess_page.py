import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
from data_preprocessing import get_preprocessing_needs_table, plot_data
from data_preprocessing import (fill_null_values, fix_inconsistent_types,
                                remove_duplicates, remove_outliers, scale_features,
                                fix_class_imbalance, interpolate_data, boxcox_transform,
                                encode_categorical_features, get_correlation_matrix)


# ---- FUNCTIONS ---- #
def plot_initial_data(plt_col):
    fig, err_msg = plot_data(st.session_state.processed_df)
    if fig is None:
        plt_col.error(err_msg)
    else:
        plt_col.write(fig)


def remove_null_values_in_data():
    st.session_state.processed_df = fill_null_values(st.session_state.processed_df)


def fix_inconsistent_types_in_data():
    st.session_state.processed_df = fix_inconsistent_types(st.session_state.processed_df)


def remove_duplicates_in_data():
    st.session_state.processed_df = remove_duplicates(st.session_state.processed_df)


def remove_outliers_in_data():
    st.session_state.processed_df = remove_outliers(st.session_state.processed_df)


def normalize_data():
    st.session_state.processed_df = scale_features(st.session_state.processed_df)


def handle_class_imbalance():
    st.session_state.processed_df = fix_class_imbalance(st.session_state.processed_df)


def make_data_periodic(periodicity):
    st.session_state.processed_df = interpolate_data(st.session_state.processed_df, periodicity)


def use_boxcox_transform():
    st.session_state.processed_df, st.session_state.box_cox_dict = boxcox_transform(st.session_state.processed_df)


def label_encode_data(cols_for_label):
    st.session_state.processed_df = encode_categorical_features(st.session_state.processed_df, cols_for_label, type="label")


def plot_correlation_matrix(col, corr_method):
    fig, err_msg = get_correlation_matrix(st.session_state.processed_df, corr_method)
    if fig is None:
        col.write(err_msg)
    else:
        col.write(fig)


def write_preprocessing_needs_table():
    preprocessing_table, progress = get_preprocessing_needs_table(st.session_state.processed_df)
    st.progress(progress, text="Data Readiness for ML Training")
    st.write(f"Readiness Score = {round(progress*100, 2)}%")
    st.write("The following table shows the preprocessing needs of your data:")
    cols = st.columns(5)

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

    st.write("---")

    second_row_cols = st.columns(5)

    second_row_cols[0].write("Data Type")
    second_row_cols[0].write(preprocessing_table.iloc[5,1][0])
    if preprocessing_table.iloc[5,1][0] == "Data suitable for Classification with insignificant class imbalance":
        second_row_cols[0].table(preprocessing_table.iloc[5,1][1])
    elif preprocessing_table.iloc[5,1][0] == "Data suitable for Classification but needs to handle class imbalance":
        # second_row_cols[0].button("Fix Imbalance", on_click=handle_class_imbalance)        
        second_row_cols[0].table(preprocessing_table.iloc[5,1][1])
        second_row_cols[0].button("Fix Imbalance", on_click=handle_class_imbalance)
    
    second_row_cols[1].write("Periodicity in Data")
    second_row_cols[1].write(preprocessing_table.iloc[6,1])
    periodicity = second_row_cols[1].selectbox("Select Periodicity in Days", options=range(1,30))
    second_row_cols[1].button("Make Data Periodic", on_click=make_data_periodic, args=(f"{periodicity}D",))

    second_row_cols[2].write("Box-Cox Transform")
    second_row_cols[2].button("Use Box-Cox Transform", on_click=use_boxcox_transform)

    second_row_cols[3].write("Label Encoding")
    cols_that_need_label_encoding = preprocessing_table.iloc[7,1]
    cols_for_label = second_row_cols[3].multiselect("Select Columns for Label Encoding", options=cols_that_need_label_encoding)
    if len(cols_for_label) > 0:
        second_row_cols[3].button("Label Encode", on_click=label_encode_data, args=(cols_for_label,))

    second_row_cols[4].write("Correlation Analysis")
    corr_method = second_row_cols[4].selectbox("Select Correlation Method", options=["pearson", "kendall", "spearman"])
    second_row_cols[4].button("Plot Correlation Matrix", on_click=plot_correlation_matrix, args=(second_row_cols[4], corr_method))

    processed_csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Processed Dataset", data=processed_csv, file_name="processed_dataset.csv", mime='text/csv', use_container_width=True)


def show_data_preprocess_page():
    # # ---- PAGE CONFIG ---- #
    # st.set_page_config(page_title="Explainable AI", layout='wide')

    # # ---- TITLE ---- #
    # st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
    # st.write('---')


    # ---- DATA PREPROCESSING ---- #
    st.subheader("Data Preprocessing")
    if st.session_state.processed_df is not None:
        _, plt_col, _ = st.columns((1,4,1))
        plot_initial_data(plt_col)
        write_preprocessing_needs_table()