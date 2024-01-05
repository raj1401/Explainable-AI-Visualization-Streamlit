import os
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as sl
import matplotlib.pyplot as plt


# ---- GLOBAL VARIABLES ---- #
TEMP_DIR = "temp_data"


@sl.cache_data
def forecast_from_classifier(_model, data_file):
    if data_file is not None:
        try:
            with NamedTemporaryFile(mode='wb', suffix=".csv", dir=TEMP_DIR, delete=False) as f:
                f.write(data_file.read())
            with open(f.name, 'r') as file:
                df = pd.read_csv(file)
        except Exception as e:
            return
    else:
        return None, None, None
    
    X, dates = df.iloc[:,1:], df.iloc[:,0].astype(str)
    y_pred = _model.predict(X)

    fig, ax = plt.subplots(figsize=(10,5))
    xticks = dates.iloc[::len(dates)//20]
    ax.plot(dates, [0.5]*len(y_pred), "--r")
    ax.scatter(dates, y_pred, s=20, label=f"Predicted values")
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0,1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90)
    ax.legend(loc='upper right')
    ax.set_ylabel("Predicted Class")
    ax.set_xlabel("Time Steps")

    ax.set_title(f"Model's Prediction on Given Data")

    df["target_value_pred"] = y_pred
    
    return fig, df, None


@sl.cache_data
def forecast_from_regressor(_model, data_file):
    if data_file is not None:
        try:
            with NamedTemporaryFile(mode='wb', suffix=".csv", dir=TEMP_DIR, delete=False) as f:
                f.write(data_file.read())
            with open(f.name, 'r') as file:
                df = pd.read_csv(file)
        except Exception as e:
            return
    else:
        return None, None, None
    
    X, dates = df.iloc[:,1:], df.iloc[:,0].astype(str)
    y_pred = _model.predict(X)

    fig, ax = plt.subplots(figsize=(10,5))
    xticks = dates.iloc[::len(dates)//20]
    ax.plot(dates, y_pred, "--b", label=f"Predicted values")
    ax.legend(loc='upper right')
    ax.set_ylabel("Predicted Values")
    ax.set_xlabel("Time Steps")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90)
    ax.set_title(f"Model's Prediction on Given Data")

    df["target_value_pred"] = y_pred
    
    return fig, df, None