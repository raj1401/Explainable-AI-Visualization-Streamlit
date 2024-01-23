import os
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as sl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import scipy.stats as stats

import numpy as np
from copy import deepcopy


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


# ----------- LSTM PREDICTION/FORECASTING ------------- #

def get_dataloader_single(df, val_fraction, batch_size, lookback, test_fraction, random_state):
    X, y = df.iloc[:,1:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, shuffle=False, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_fraction, shuffle=False, random_state=random_state)

    y_train = np.array(y_train).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)

    train_feat, train_target = [], []
    for i in range(len(y_train_scaled) - lookback):
        f = y_train_scaled[i:i+lookback]
        t = y_train_scaled[i+1:i+lookback+1]
        train_feat.append(f)
        train_target.append(t)
    
    val_feat, val_target = [], []
    for i in range(len(y_val_scaled) - lookback):
        f = y_val_scaled[i:i+lookback]
        t = y_val_scaled[i+1:i+lookback+1]
        val_feat.append(f)
        val_target.append(t)
    
    train_dataloader = DataLoader(TensorDataset(torch.Tensor(train_feat), torch.Tensor(train_target)), shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(TensorDataset(torch.Tensor(val_feat), torch.Tensor(val_target)), shuffle=True, batch_size=batch_size)
    
    return train_dataloader, val_dataloader


def get_dataloader_multiple(y_list, val_fraction):
    batch_size = len(y_list)
    max_length = min(len(y_t) for y_t in y_list)

    data = np.empty((batch_size, max_length))

    for i in range(batch_size):
        y_t = np.array(y_list[i][:max_length])
        data[i] = (y_t - np.mean(y_t)) / np.std(y_t)
    
    train_length = int((1 - val_fraction) * max_length)
    train_data = data[:, :train_length]
    val_data = data[:, train_length+1:]

    train_feat, train_target = train_data[:, :-1].reshape(batch_size, -1, 1), train_data[:, 1:].reshape(batch_size, -1, 1)
    val_feat, val_target = val_data[:, :-1].reshape(batch_size, -1, 1), val_data[:, 1:].reshape(batch_size, -1, 1)

    train_dataloader = DataLoader(TensorDataset(torch.Tensor(train_feat), torch.Tensor(train_target)), shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(TensorDataset(torch.Tensor(val_feat), torch.Tensor(val_target)), shuffle=True, batch_size=batch_size)
    
    return train_dataloader, val_dataloader


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.fc(out[:, -1, :])  # Take the output of the last time step
        out = self.fc(out)
        return out


def train(dataloader:DataLoader, model:myLSTM, loss_fn, optimizer, device):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # Computing prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 10 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}", end="\r")
    
    print(f"loss: {loss:>7f}")


def validate(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")
    return test_loss


def predict_from_LSTM(model:myLSTM, initial_data, future_steps):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(initial_data.reshape(-1,1))
    outputs = list(scaled_data.flatten())

    with torch.no_grad():
        for _ in range(future_steps):
            scaled_data = torch.Tensor(np.array(outputs).reshape(-1,1))
            pred = model(scaled_data)
            outputs.append(pred[-1,0])

    rescaled_data = scaler.inverse_transform(np.array(outputs).reshape(-1,1))
    return rescaled_data.flatten()

# --- REGRESSOR

def train_LSTM_regressor(df, y_list, from_df, val_fraction, batch_size, lookback, input_size,
                         hidden_size, num_layers, output_size, num_epochs, learning_rate,
                         test_fraction, random_state):
    
    # Getting the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    if from_df:
        train_dataloader, val_dataloader = get_dataloader_single(df, val_fraction, batch_size, lookback,
                                                                 test_fraction, random_state)
    else:
        train_dataloader, val_dataloader = get_dataloader_multiple(y_list, val_fraction)

    # # Hyperparameters
    # input_size = 1  # Assuming univariate time series data
    # output_size = 1
    # learning_rate = 0.001
    # num_epochs = 100

    model = myLSTM(input_size, hidden_size, num_layers, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} \n ----------------------------")

        train(train_dataloader, model, criterion, optimizer, device)
        val_loss = validate(val_dataloader, model, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_state)
    model.eval()
    return model


def lstm_regression_forecasting(df, test_fraction, random_state, lstm_model=None, y_list=None, from_df=True,
                                val_fraction=0.2, batch_size=32, lookback=6, input_size=1, hidden_size=64,
                                num_layers=2, output_size=1, num_epochs=100, learning_rate=0.001, future_steps=100):
    try:
        # Cached Model
        if lstm_model is None:
            trained_model = train_LSTM_regressor(
                df=df,
                y_list=y_list,
                from_df=from_df,
                val_fraction=val_fraction,
                batch_size=batch_size,
                lookback=lookback,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                test_fraction=test_fraction,
                random_state=random_state
            )
        else:
            trained_model = lstm_model

        if from_df:
            train_length = int((1-test_fraction)*len(df.iloc[:,-1]))
            train_series = np.array(df.iloc[:train_length,-1])
            predictions = predict_from_LSTM(trained_model, train_series, future_steps+(len(df.iloc[:,-1]) - train_length))

            # Calculating the t-statistics
            testing_portion = df.iloc[train_length:, -1]
            testing_pred = predictions[train_length:len(df.iloc[:,-1])]
            CONFIDENCE_LEVEL = 0.95
            residuals = np.array(testing_portion - testing_pred)
            residual_std = residuals.std()

            t_stat = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=len(residuals))
            lower_bound = testing_pred - t_stat * residual_std
            upper_bound = testing_pred + t_stat * residual_std

            fig, ax = plt.subplots(figsize=(10,5))

            y_min = predictions.min()
            y_max = predictions.max()

            x_vals = list(range(len(predictions)))
            original_x_vals = x_vals[:len(df.iloc[:,-1])]
            ax.plot(original_x_vals, df.iloc[:,-1], "b", label="Original Time Series Data")
            ax.plot(x_vals, predictions, "--y", label="LSTM Forecasting")
            ax.fill_between(original_x_vals[train_length:], lower_bound, upper_bound, alpha=0.2, label=f'{CONFIDENCE_LEVEL*100}% CI')
            ax.axvline(x=train_length, color="black")
            ax.axvline(x=len(df.iloc[:,-1]), color="black")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(df.columns[-1])
            ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.05))
            ax.set_title("Forecasting using LSTM")
            ax.text(train_length//4, y_max + 0.075*(y_max-y_min), 'Training and Validation Set', verticalalignment='center', fontsize=12)
            ax.text(train_length+(len(df.iloc[:,-1])-train_length)//6, y_max + 0.075*(y_max-y_min), 'Testing Set', verticalalignment='center', fontsize=12)
            ax.text(len(df.iloc[:,-1])+(len(predictions)-len(df.iloc[:,-1]))//3, y_max + 0.075*(y_max-y_min), 'Forecasting', verticalalignment='center', fontsize=12)
            return fig, None, trained_model
        else:
            train_length = int((1-test_fraction)*len(y_list[0]))
            train_series = np.array(y_list[0].iloc[:train_length])
            predictions = predict_from_LSTM(trained_model, train_series, future_steps+(len(y_list[0]) - train_length))

            # Calculating the t-statistics
            testing_portion = y_list[0].iloc[train_length:]
            testing_pred = predictions[train_length:len(y_list[0])]
            CONFIDENCE_LEVEL = 0.95
            residuals = np.array(testing_portion - testing_pred)
            residual_std = residuals.std()

            t_stat = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=len(residuals))
            lower_bound = testing_pred - t_stat * residual_std
            upper_bound = testing_pred + t_stat * residual_std

            fig, ax = plt.subplots(figsize=(10,5))

            y_min = predictions.min()
            y_max = predictions.max()

            x_vals = list(range(len(predictions)))
            original_x_vals = x_vals[:len(y_list[0])]
            ax.plot(original_x_vals, y_list[0], "b", label="Original Time Series Data")
            ax.plot(x_vals, predictions, "--y", label="LSTM Forecasting")
            ax.fill_between(original_x_vals[train_length:], lower_bound, upper_bound, alpha=0.2, label=f'{CONFIDENCE_LEVEL*100}% CI')
            ax.axvline(x=train_length, color="black")
            ax.axvline(x=len(y_list[0]), color="black")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(y_list[0].name)
            ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.05))
            ax.set_title("Forecasting for the first series using LSTM")
            ax.text(train_length//4, y_max + 0.075*(y_max-y_min), 'Training and Validation Set', verticalalignment='center', fontsize=12)
            ax.text(train_length+(len(y_list[0])-train_length)//6, y_max + 0.075*(y_max-y_min), 'Testing Set', verticalalignment='center', fontsize=12)
            ax.text(len(y_list[0])+(len(predictions)-len(y_list[0]))//3, y_max + 0.075*(y_max-y_min), 'Forecasting', verticalalignment='center', fontsize=12)
            return fig, None, trained_model
    except Exception as e:
        return None, e, None

