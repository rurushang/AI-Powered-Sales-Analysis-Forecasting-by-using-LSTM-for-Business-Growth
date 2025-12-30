import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import joblib
import os

# -----------------------------
# File and directory configuration
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
DATA_PATH = r'D:\YZU\763  AI\AI-Powered-Sales-Analysis-Forecasting-for-Business-Growth\final_data.csv'

# -----------------------------
# Load model and scaler
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Train/test split function
# -----------------------------
def train_test_split(df, train_end, test_end):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    train_end = pd.Timestamp(train_end)
    test_end = pd.Timestamp(test_end)
    train_set = df[df.index <= train_end]
    test_set = df[(df.index > train_end) & (df.index <= test_end)]
    return train_set, test_set

# -----------------------------
# Lookback function for LSTM
# -----------------------------
def lookback(df, window):
    X, Y = [], []
    for i in range(window, len(df)):
        X.append(df[i-window:i, 0])
        Y.append(df[i,0])
    return np.array(X), np.array(Y)

# -----------------------------
# Forecast future function
# -----------------------------
def forecast_future(model, last_window, n_days):
    future_scaled = []
    current_batch = last_window[-window:].reshape(1, window, 1)
    for _ in range(n_days):
        pred = model.predict(current_batch)
        future_scaled.append(pred[0,0])
        current_batch = np.append(current_batch[:,1:,:], [[pred[0]]], axis=1)
    return np.array(future_scaled)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Sales Forecasting using LSTM")

# 1️⃣ Load CSV and automatically detect separator
# Read CSV normally
try:
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# If column names are unknown — show them
st.write("Detected columns:", list(data.columns))

# Try automatic detection of date + value
possible_date_cols = [c for c in data.columns if 'date' in c.lower()]
if len(possible_date_cols) == 0:
    possible_date_cols = [data.columns[0]]  # fallback: first col

date_col = possible_date_cols[0]
value_col = [c for c in data.columns if c != date_col][0]

data = data[[date_col, value_col]]
data.columns = ['date', 'total_amount']

data.set_index('date', inplace=True)
data.index = pd.to_datetime(data.index, errors='coerce')

# Drop rows where date conversion failed
data.dropna(subset=['total_amount'], inplace=True)

st.write("Data preview:")
st.write(data.head())

# 2️⃣ Split train/test
train_end = '2018-04-30'
test_end = '2018-08-29'
train_df, test_df = train_test_split(data, train_end, test_end)

if len(train_df)==0 or len(test_df)==0:
    st.error(f"Train/Test set is empty. Data min: {data.index.min()}, max: {data.index.max()}")
    st.stop()
else:
    st.write(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# 3️⃣ Scale data
value_col = 'total_amount'
train_scaled = scaler.fit_transform(train_df[[value_col]])
test_scaled = scaler.transform(test_df[[value_col]])

# 4️⃣ Lookback
window = 1
X_train, y_train = lookback(train_scaled, window)
X_test, y_test = lookback(test_scaled, window)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 5️⃣ Predict on test set
y_pred = model.predict(X_test)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_actual = scaler.inverse_transform(y_pred)

# 6️⃣ Forecast future
forecast_days = st.number_input("Enter number of days to forecast:", min_value=1, value=30, max_value=150)
future_scaled = forecast_future(model, test_scaled[-window:], forecast_days)
future_actual = scaler.inverse_transform(future_scaled.reshape(-1,1))
extended_index = pd.date_range(start=test_df.index[-1]+pd.Timedelta(days=1), periods=forecast_days)

# 7️⃣ Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_df.index, y=y_test_actual.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=test_df.index, y=y_pred_actual.flatten(), mode='lines', name='Predicted'))
fig.add_trace(go.Scatter(x=extended_index, y=future_actual.flatten(), mode='lines', name='Forecast', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted vs Forecast',
                  xaxis_title='Date', yaxis_title=value_col, legend_title='Legend')
st.plotly_chart(fig)
# Export actual vs predicted
# EXPORT: Align length to match LSTM output
aligned_dates = test_df.index[window:]   # y_pred starts from the 2nd day

export_test = pd.DataFrame({
    "date": aligned_dates,
    "actual": y_test_actual.flatten(),
    "predicted": y_pred_actual.flatten()
})
export_test.to_csv("test_actual_predicted.csv", index=False)

export_forecast = pd.DataFrame({
    "date": extended_index,
    "forecast": future_actual.flatten()
})
export_forecast.to_csv("forecast_future.csv", index=False)
