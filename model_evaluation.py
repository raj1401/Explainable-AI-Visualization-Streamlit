import os

import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_train, y_train, ax=ax, name='Training')
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name='Testing')

        return fig, None
    except Exception as e:
        return None, e


def plot_roc_auc(df, model, random_state, test_fraction):
    try:    
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax, name='Training')
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name='Testing')

        return fig, None
    except Exception as e:
        return None, e

def plot_confusion_matrix(df, model, random_state, test_fraction):
    try:    
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, 
                                            display_labels=['Non Detect', 'Detect'],
                                            ax=ax, cmap=plt.cm.Blues, colorbar=False)

        return fig, None
    except Exception as e:
        return None, e

def get_classification_time_series_predictions(df, model, random_state, test_fraction):
    try:    
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
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
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        fig, ax = plt.subplots()
        PredictionErrorDisplay.from_estimator(model, X_test, y_test, ax=ax)

        return fig, None
    except Exception as e:
        return None, e

def get_regression_metrics(df, model, random_state, test_fraction):
    try:    
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
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

def get_regression_time_series_predictions(df, model, random_state, test_fraction):
    try:    
        dates = df.iloc[:,0].astype(str)
        X, y = df.iloc[:,1:-1], df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)

        # Calculating the t-statistics
        CONFIDENCE_LEVEL = 0.95
        y_pred = model.predict(X_test)
        residuals = np.array(y_test - y_pred)
        residual_std = residuals.std()

        t_stat = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=len(residuals))
        lower_bound = y_pred - t_stat * residual_std
        upper_bound = y_pred + t_stat * residual_std

        # Plotting
        target_name = df.columns[-1]
        fig, ax = plt.subplots(figsize=(10,6))   

        x_vals = list(dates)
        x_vals_train = x_vals[:len(y_train)]
        x_vals_test = x_vals[len(y_train):]

        y_min = y.min()
        y_max = y.max()
        
        x_ticks = dates.iloc[::len(dates)//20]

        ax.plot(x_vals_train, y_train, "-o", label=f"Training {target_name} values")

        ax.plot(x_vals_test, y_test, "-gx", label=f"Target {target_name} values")
        ax.plot(x_vals_test, y_pred, "--r", label=f"Predicted {target_name} values")

        ax.fill_between(x_vals_test, lower_bound, upper_bound, alpha=0.2, label=f'{CONFIDENCE_LEVEL*100}% CI')

        ax.axvline(x=len(y_train), color='black', linestyle='--')
        ax.text(0.15*len(y), y_min - 0.25*(y_max-y_min), 'Training and Validation Set', verticalalignment='center', fontsize=12)
        ax.text(0.85*len(y), y_min - 0.25*(y_max-y_min), 'Testing Set', verticalalignment='center', fontsize=12)

        ax.legend(loc="upper right")
        
        ax.set_ylim(bottom=y_min - 0.5*(y_max-y_min), top=y_max + 0.5*(y_max-y_min))
        ax.set_ylabel(target_name)
        ax.set_xlabel("Time Steps")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=90)
        ax.set_title(f"Model's Performance on Predicting {target_name} values \n from Selected Features")
        
        return fig, None
    except Exception as e:
        return None, e