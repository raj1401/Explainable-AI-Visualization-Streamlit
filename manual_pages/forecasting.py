import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
from helper_functions import data_file_loader, multi_time_series_loader, delete_folder_contents
from model_forecasting import forecast_from_classifier, forecast_from_regressor, lstm_regression_forecasting


# ---- GLOBAL VARIABLES ---- #
TEMP_DIR = "temp_data"


# ---- FUNCTIONS ---- #
def reset_lstm_model():
    st.session_state.LSTM_Model = None


def plot_lstm_forecasts_single(input_data_file, future_steps, plot_col, batch_size, lookback, hidden_size, num_layers):
    input_df = data_file_loader(input_data_file, temp_dir=TEMP_DIR)
    fig, err_msg, trained_lstm = lstm_regression_forecasting(df=input_df, test_fraction=st.session_state.TEST_FRACTION,
                                                            random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, from_df=True,
                                                            future_steps=future_steps, lstm_model=st.session_state.LSTM_Model,
                                                            batch_size=batch_size, lookback=lookback, hidden_size=hidden_size,
                                                            num_layers=num_layers)
    st.session_state.LSTM_Model = trained_lstm
    if fig is None:
        plot_col.write(err_msg)
    else:
        plot_col.write(fig)


def plot_lstm_forecasts_multiple(input_data_file, future_steps, plot_col, batch_size, hidden_size, num_layers):
    y_list = multi_time_series_loader(input_data_file, temp_dir=TEMP_DIR)
    fig, err_msg, trained_lstm = lstm_regression_forecasting(df=None, y_list=y_list, test_fraction=st.session_state.TEST_FRACTION,
                                                            random_state=st.session_state.TRAIN_TEST_RANDOM_STATE, from_df=False,
                                                            future_steps=future_steps, lstm_model=st.session_state.LSTM_Model,
                                                            batch_size=batch_size, hidden_size=hidden_size,
                                                            num_layers=num_layers)
    st.session_state.LSTM_Model = trained_lstm
    if fig is None:
        plot_col.write(err_msg)
    else:
        plot_col.write(fig)


def plot_forecasts(new_data, left_col, right_col):
    if st.session_state.model_type in st.session_state.classification_models:
        fig, df, err_msg = forecast_from_classifier(_model=st.session_state.model_on_selected_feats, data_file=new_data)
        if err_msg is not None:
            st.write(err_msg)
        else:
            left_col.write(fig)
            csv_file = df.to_csv(index=False).encode('utf-8')
            right_col.write("You can download the augmented dataset where the last column contains the predicted values:")
            with right_col:
                st.download_button(label="Download Augmented Dataset", data=csv_file, file_name="augmented_dataset.csv", mime='text/csv')

    elif st.session_state.model_type in st.session_state.regression_models:
        fig, df, err_msg = forecast_from_regressor(_model=st.session_state.model_on_selected_feats, data_file=new_data)
        if err_msg is not None:
            st.write(err_msg)
        else:
            left_col.write(fig)
            csv_file = df.to_csv(index=False).encode('utf-8')
            right_col.write("You can download the augmented dataset where the last column contains the predicted values:")
            with right_col:
                st.download_button(label="Download Augmented Dataset", data=csv_file, file_name="augmented_dataset.csv", mime='text/csv')


def show_forecasting_page():
    # # ---- PAGE CONFIG ---- #
    # st.set_page_config(page_title="Explainable AI", layout='wide')

    # # ---- TITLE ---- #
    # st.markdown("<h1 style='text-align: center;'> DPI - ML Platform </h1>", unsafe_allow_html=True)
    # st.write('---')

    # ---- FORECASTING ---- #
    with st.container():
        st.markdown("<h2 style='text-align: center;'> Forecasting From The Data </h2>", unsafe_allow_html=True)
        st.write("""
                    Forecasting from any provided data can be a univariate or a multivariate problem. In the univariate
                    setting, we try to predict the future values of a time-series using only the values it takes in the
                    previous time steps. We accomplish this usually by training a Recurrent Neural Network.
                    On the other hand, in the multivariate setting, we try to predict the future
                    values of a time-series based on other independent variables whose values are/will be known in the future.
        """)
        if st.session_state.model_on_selected_feats is not None:
            problem_type = st.selectbox("Problem Type", options=["Univariate", "Multivariate"], on_change=reset_lstm_model)

            if problem_type == "Univariate":
                st.write("""We can train an LSTM, a special kind of Recurrent Neural Network (RNN) to predict future values
                        from a periodic time-series data. Do you want to upload a DataFrame (CSV File) to predict future values of the
                        time series it contains in its last column, or do you want to upload a DataFrame (CSV File) that contains
                        multiple time-series data of the same target variable and same periodicity in its columns and train the LSTM on them?
                        The latter is recommended if you have 100s of time-series data.""")
                input_type = st.selectbox("Input Type", options=["Single DataFrame", "Multiple Data Series"], on_change=reset_lstm_model)
                
                st.write("How many time steps in the future do you want to make the predictions?")
                f_col, _ = st.columns((1,3))
                future_steps = f_col.number_input("Future Time Steps", min_value=20, step=1)
                st.write("Model and Training Hyperparameters")
                col1, col2, col3, col4 = st.columns(4)
                batch_size = col1.number_input("Batch Size", min_value=4, step=1, on_change=reset_lstm_model)
                hidden_size = col2.number_input("LSTM Hidden State Size", min_value=8, step=1, on_change=reset_lstm_model)
                num_layers = col3.number_input("Number of Hidden LSTM Layers", min_value=1, step=1, on_change=reset_lstm_model)

                if input_type == "Single DataFrame":
                    lookback = col4.number_input("Window Size", min_value=1, step=1, on_change=reset_lstm_model)
                    single_df_data = st.file_uploader("Single DataFrame", type="csv")
                    if single_df_data is not None:
                        _, plot_col, _ = st.columns((1,4,1))
                        # sl.session_state.LSTM_plot_col = plot_col
                        # plot_col.button("Train LSTM Model", use_container_width=True, on_click=plot_lstm_forecasts_single, args=(single_df_data, future_steps, plot_col, batch_size, lookback, hidden_size, num_layers))
                        plot_lstm_forecasts_single(input_data_file=single_df_data, future_steps=future_steps, plot_col=plot_col,
                                                batch_size=batch_size, lookback=lookback, hidden_size=hidden_size, num_layers=num_layers)
                
                if input_type == "Multiple Data Series":
                    multi_df_data = st.file_uploader("Multiple Data Series", type="csv")
                    if multi_df_data is not None:
                        _, plot_col, _ = st.columns((1,4,1))
                        # sl.session_state.LSTM_plot_col = plot_col
                        # plot_col.button("Train LSTM Model", use_container_width=True, on_click=plot_lstm_forecasts_multiple, args=(multi_df_data, future_steps, plot_col, batch_size, hidden_size, num_layers))
                        plot_lstm_forecasts_multiple(input_data_file=multi_df_data, future_steps=future_steps, plot_col=plot_col,
                                                    batch_size=batch_size, hidden_size=hidden_size, num_layers=num_layers)

                # if new_data is not None:
                #     _, plot_col, _ = sl.columns((1,4,1))
                #     plot_lstm_forecasts(input_type=input_type, input_data_file=new_data, future_steps=future_steps, plot_col=plot_col)

            if problem_type == "Multivariate":
                # Getting New Data
                st.write("""You can now use the above trained model to make predictions
                            about the target variable using your own data. Make sure that the data that you upload only has
                            the features that were chosen above to train the final model (the first column should
                            have the time the data points were recorded).""")
                # sl.markdown("""<h5 style='text-align: center;'> You can now use the above trained model to make predictions
                #             about the target variable using your own data. Make sure that the data that you upload only has
                #             the features that were chosen above to train the final model (the first column should
                #             have the time the data points were recorded). </h5>""", unsafe_allow_html= True)

                new_data = st.file_uploader("New Input Data", type="csv")
                if new_data is not None:
                    left_graph_col, right_col = st.columns((3,1))
                    plot_forecasts(new_data, left_graph_col, right_col)

    delete_folder_contents(TEMP_DIR)