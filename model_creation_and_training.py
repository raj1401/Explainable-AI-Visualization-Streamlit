import os
from tempfile import NamedTemporaryFile
from helper_functions import delete_folder_contents

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import streamlit as sl


# ---- GLOBAL VARIABLES ---- #
TEMP_DIR = "temp_data"
TEST_FRACTION = 0.2
NUM_ITERS = 3
CROSS_VALID = 5
RANDOM_STATE = 123


if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


# ---------- RANDOM FOREST CLASSIFIER ---------- #
    
@sl.cache_resource
def create_and_search_tree_classifier(df, **kwargs):
    """
    This function performs 5-fold cross-validation and randomly searches over parameters
    to find out optimum parameters for the random forest classifier and returns
    the optimum parameters and models
    """

    base_classifier = CatBoostClassifier(loss_function="Logloss")

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    param_distributions = kwargs['param_distributions']

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        base_classifier,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_classifier(df:pd.DataFrame, **kwargs):    
    param_distributions = kwargs['param_distributions']
    param_distributions["loss_function"] = "Logloss"

    classifier = CatBoostClassifier(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    classifier.fit(X=X_train, y=y_train)

    return classifier, RANDOM_STATE, TEST_FRACTION


# ---------- RANDOM FOREST REGRESSOR ---------- #

@sl.cache_resource
def create_and_search_tree_regressor(df, **kwargs):
    """
    This function performs 5-fold cross-validation and randomly searches over parameters
    to find out optimum parameters for the random forest regressor and returns
    the optimum parameters and models
    """

    base_regressor = CatBoostRegressor()

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    param_distributions = kwargs['param_distributions']

    # scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        base_regressor,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_regressor(df:pd.DataFrame, **kwargs):    
    param_distributions = kwargs['param_distributions']

    regressor = CatBoostRegressor(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    regressor.fit(X=X_train, y=y_train)
    return regressor, RANDOM_STATE, TEST_FRACTION


# ---------- LOGISTIC REGRESSION ------------ #

@sl.cache_resource
def create_and_search_logistic_regression(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(penalty='elasticnet', solver='saga'))
    ])

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_logistic_regression(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(penalty='elasticnet', solver='saga'))
    ])

    pipeline.set_params(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    pipeline.fit(X=X_train, y=y_train)

    return pipeline, RANDOM_STATE, TEST_FRACTION


# ---------- LINEAR REGRESSION -------------- #

@sl.cache_resource
def create_and_search_linear_regression(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', ElasticNet())
    ])

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_linear_regression(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', ElasticNet())
    ])

    pipeline.set_params(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    pipeline.fit(X=X_train, y=y_train)

    return pipeline, RANDOM_STATE, TEST_FRACTION



# ---------- KNN CLASSIFIER ----------------- #

@sl.cache_resource
def create_and_Search_KNN_classifer(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn_classifier', KNeighborsClassifier())
    ])

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_KNN_classifier(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn_classifier', KNeighborsClassifier())
    ])

    pipeline.set_params(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    pipeline.fit(X=X_train, y=y_train)

    return pipeline, RANDOM_STATE, TEST_FRACTION


# ---------- KNN REGRESSOR ----------------- #

@sl.cache_resource
def create_and_Search_KNN_regressor(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn_regressor', KNeighborsRegressor())
    ])

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_KNN_regressor(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn_regressor', KNeighborsRegressor())
    ])

    pipeline.set_params(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    pipeline.fit(X=X_train, y=y_train)

    return pipeline, RANDOM_STATE, TEST_FRACTION


# ---------- SVM CLASSIFIER ------------------ #

@sl.cache_resource
def create_and_Search_SVM_classifer(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_classifier', SVC())
    ])

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_SVM_classifier(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_classifier', SVC(probability=False))
    ])

    pipeline.set_params(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=True, stratify=y, random_state=RANDOM_STATE)

    pipeline.fit(X=X_train, y=y_train)

    return pipeline, RANDOM_STATE, TEST_FRACTION


# ---------- SVM REGRESSOR ------------------ #

@sl.cache_resource
def create_and_Search_SVM_regressor(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_regressor', SVR())
    ])

    scoring = make_scorer(lambda y_true, y_pred: (f1_score(y_true, y_pred, pos_label=0) + f1_score(y_true, y_pred, pos_label=1)) / 2)

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=NUM_ITERS,
        cv=CROSS_VALID,
        n_jobs=-1,
        scoring=scoring,
        random_state=42,
        verbose=0
    )

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_params, best_model


@sl.cache_resource
def train_final_SVM_regressor(df, **kwargs):
    param_distributions = kwargs['param_distributions']

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_regressor', SVR())
    ])

    pipeline.set_params(**param_distributions)

    dates = df.iloc[:,0].astype(str)
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_FRACTION, shuffle=False, random_state=RANDOM_STATE)

    pipeline.fit(X=X_train, y=y_train)

    return pipeline, RANDOM_STATE, TEST_FRACTION



# ---------- LEAD-LAG CORRELATIONS ---------- #

@sl.cache_data
def get_lead_lag_correlations(df:pd.DataFrame):
    df_copy = df.copy(deep=True)
    X, y = df_copy.iloc[:,:-1], df_copy.iloc[:,-1].to_frame()

    Is = range(-3, 3)
    dfs = pd.DataFrame()

    for i in Is:
        X_shifted = X.shift(i)
        X_shifted['target_class'] = y
        dfs[i] = X_shifted.corr(method='spearman')['target_class']
    
    dfs_T = dfs.iloc[:-1,:].T
    correlations = pd.DataFrame()
    correlations['Lags'] = dfs_T.idxmax()
    correlations['values'] = dfs_T.max()

    lead_corr = correlations[[correlations['values'] >= 0] and correlations['Lags'] <= 0]  # With only lag time
    lag_corr = correlations[correlations['values'] >= 0]   # with lead and lag time both

    return lead_corr, lag_corr


@sl.cache_data
def shift_dataframe(df:pd.DataFrame, corr:pd.DataFrame):
    lags = corr['Lags']
    to_shift = corr.index.values
    df_shift = df.copy(deep=True)
    
    for col in to_shift:
        shift_by = lags[col].astype('int')
        df_shift[col] = df_shift[col].shift(shift_by, fill_value=0)
        
    return df_shift


