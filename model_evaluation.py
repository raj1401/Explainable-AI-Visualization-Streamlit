import os

import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import PredictionErrorDisplay, explained_variance_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tempfile import NamedTemporaryFile
from helper_functions import delete_folder_contents


temp_dir = "temp_data"

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


# ------------- SHAP PLOTS -------------- #

def plot_shap_bar(shap_vals):
    try:
        fig = plt.figure()
        _ = shap.plots.bar(shap_vals)

        return fig, None
    except Exception as e:
        return None, e


def plot_shap_beeswarm(shap_vals):
    try:
        fig = plt.figure()
        _ = shap.plots.beeswarm(shap_vals)

        return fig, None
    except Exception as e:
        return None, e
    

def plot_shap_heatmap(shap_vals):
    try:
        fig = plt.figure()
        _ = shap.plots.heatmap(shap_vals)

        return fig, None
    except Exception as e:
        return None, e


# --------------- RECURSIVE FEATURE ELIMINATION ---------------- #




# --------------- CLASSIFIER PLOTS ---------------- #

def plot_precision_recall(df, model, random_state, test_fraction):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_train, y_train, ax=ax, name='Training')
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name='Testing')

        return fig, None
    except Exception as e:
        return None, e


def plot_roc_auc(df, model, random_state, test_fraction):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax, name='Training')
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name='Testing')

        return fig, None
    except Exception as e:
        return None, e

def plot_confusion_matrix(df, model, random_state, test_fraction):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, 
                                            display_labels=['Non Detect', 'Detect'],
                                            ax=ax, cmap=plt.cm.Blues, colorbar=False)

        return fig, None
    except Exception as e:
        return None, e

def get_classification_time_series_predictions(df, model, random_state, test_fraction, dates):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        target_name = df.columns[-1]
        fig, ax = plt.subplots(figsize=(10,5))
        y_pred = model.predict(X_test)

        x_vals = range(len(y_test))
        x_ticks = dates.iloc[len(y_train):]

        diff = y_pred - y_test

        ax.plot(x_vals, [0]*len(y_test), "--r")
        ax.scatter(x_vals, diff, s=20, label=f"(Predicted - Target) {target_name} values")
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1,0,1])
        ax.legend(loc='upper right')
        ax.set_ylabel(target_name)
        ax.set_xlabel("Time Steps")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_ticks, rotation=90)
        ax.set_title(f"Model's Performance on Predicting {target_name} values \n from Selected Features")
        
        return fig, None
    except Exception as e:
        return None, e


# ------------ REGRRESSOR PLOTS ------------- #

def plot_prediction_error(df, model, random_state, test_fraction):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        PredictionErrorDisplay.from_estimator(model, X_test, y_test, ax=ax)

        return fig, None
    except Exception as e:
        return None, e

def get_regression_metrics(df, model, random_state, test_fraction):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)
        test_pred = model.predict(X_test)

        exp_var_score = explained_variance_score(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)
        r2_scr = r2_score(y_test, test_pred)

        exp_var_score = round(exp_var_score, 3)
        mse = round(mse, 3)
        r2_scr = round(r2_scr, 3)

        return exp_var_score, mse, r2_scr, None
    except Exception as e:
        return None, None, None, e

def get_regression_time_series_predictions(df, model, random_state, test_fraction, dates):
    try:    
        X, y = df.iloc[:,:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        target_name = df.columns[-1]
        fig, ax = plt.subplots(figsize=(10,5))
        y_pred = model.predict(X_test)

        x_vals = range(len(y_test))
        x_ticks = dates.iloc[len(y_train):]

        ax.plot(x_vals, y_test, "-o", label=f"Target {target_name} values")
        ax.plot(x_vals, y_pred, "--r", label=f"Predicted {target_name} values")
        ax.legend()
        ax.set_ylabel(target_name)
        ax.set_xlabel("Time Steps")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_ticks, rotation=90)
        ax.set_title(f"Model's Performance on Predicting {target_name} values \n from Selected Features")
        
        return fig, None
    except Exception as e:
        return None, e