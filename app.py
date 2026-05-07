# ============================================================
# STREAMLIT DASHBOARD
# WALMART MULTI-SERIES FORECASTING
# ============================================================

# Run with:
# streamlit run app.py

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Walmart Forecast Dashboard",
    layout="wide"
)

st.title("Walmart Multi-Series Forecasting Dashboard")

st.markdown("""
This dashboard compares:

- Exponential Smoothing
- ARIMA
- Random Forest

for Walmart department sales forecasting.
""")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("train_2.csv")

    df['Date'] = pd.to_datetime(df['Date'])

    return df


df = load_data()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Dashboard Controls")

store_id = st.sidebar.selectbox(
    "Select Store",
    sorted(df['Store'].unique())
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=4,
    max_value=26,
    value=12
)

selected_depts = st.sidebar.multiselect(
    "Select Departments",
    sorted(df['Dept'].unique()),
    default=[1, 2, 3]
)

target_series = st.sidebar.selectbox(
    "Select Target Series",
    options=["Total_Sales"] + [f"Dept_{d}" for d in selected_depts]
)


# ============================================================
# PREPARE DATA
# ============================================================

df_store = df[df['Store'] == store_id]

weekly_sales = (
    df_store.groupby(['Date', 'Dept'])['Weekly_Sales']
    .sum()
    .reset_index()
)

sales_pivot = (
    weekly_sales[weekly_sales['Dept'].isin(selected_depts)]
    .pivot(index='Date', columns='Dept', values='Weekly_Sales')
)

sales_pivot.columns = [f"Dept_{c}" for c in sales_pivot.columns]

sales_pivot = sales_pivot.fillna(0)

sales_pivot['Total_Sales'] = sales_pivot.sum(axis=1)


# ============================================================
# SHOW DATA
# ============================================================

st.subheader("Sales Data")

st.dataframe(sales_pivot.head())


# ============================================================
# HISTORICAL SERIES PLOT
# ============================================================

st.subheader("Historical Sales")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(
    sales_pivot.index,
    sales_pivot[target_series],
    label=target_series
)

ax.set_title(f"{target_series} Historical Sales")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

train = sales_pivot.iloc[:-forecast_horizon]
test = sales_pivot.iloc[-forecast_horizon:]


# ============================================================
# EXPONENTIAL SMOOTHING
# ============================================================

ets_model = ExponentialSmoothing(
    train[target_series],
    trend='add',
    seasonal='add',
    seasonal_periods=52
).fit()

ets_forecast = ets_model.forecast(forecast_horizon)


# ============================================================
# ARIMA
# ============================================================

arima_model = ARIMA(
    train[target_series],
    order=(1,1,1)
).fit()

arima_forecast = arima_model.forecast(forecast_horizon)


# ============================================================
# RANDOM FOREST
# ============================================================

def create_features(series):

    df_feat = pd.DataFrame()

    df_feat['y'] = series

    df_feat['lag_1'] = df_feat['y'].shift(1)
    df_feat['lag_2'] = df_feat['y'].shift(2)
    df_feat['lag_4'] = df_feat['y'].shift(4)
    df_feat['lag_12'] = df_feat['y'].shift(12)

    df_feat['roll_mean_4'] = (
        df_feat['y'].rolling(4).mean()
    )

    df_feat['roll_mean_12'] = (
        df_feat['y'].rolling(12).mean()
    )

    return df_feat.dropna()


feature_df = create_features(train[target_series])

X_train = feature_df.drop(columns='y')
y_train = feature_df['y']

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train)


history = train[target_series].copy()

rf_predictions = []

for i in range(forecast_horizon):

    temp = pd.DataFrame({

        'lag_1': [history.iloc[-1]],
        'lag_2': [history.iloc[-2]],
        'lag_4': [history.iloc[-4]],
        'lag_12': [history.iloc[-12]],

        'roll_mean_4': [
            history.iloc[-4:].mean()
        ],

        'roll_mean_12': [
            history.iloc[-12:].mean()
        ]
    })

    pred = rf_model.predict(temp)[0]

    rf_predictions.append(pred)

    history.loc[
        history.index[-1] + pd.Timedelta(weeks=1)
    ] = pred


# ============================================================
# FORECAST PLOT
# ============================================================

st.subheader("Forecast Comparison")

fig, ax = plt.subplots(figsize=(15,6))

ax.plot(
    train.index,
    train[target_series],
    label='Train'
)

ax.plot(
    test.index,
    test[target_series],
    label='Test'
)

ax.plot(
    test.index,
    ets_forecast,
    label='ETS Forecast'
)

ax.plot(
    test.index,
    arima_forecast,
    label='ARIMA Forecast'
)

ax.plot(
    test.index,
    rf_predictions,
    label='RF Forecast'
)

ax.set_title(f"Forecast Comparison - {target_series}")

ax.legend()

st.pyplot(fig)


# ============================================================
# ACCURACY METRICS
# ============================================================

st.subheader("Model Accuracy")

actual = test[target_series]

results = pd.DataFrame({

    'Model': ['ETS', 'ARIMA', 'Random Forest'],

    'MAE': [

        mean_absolute_error(actual, ets_forecast),

        mean_absolute_error(actual, arima_forecast),

        mean_absolute_error(actual, rf_predictions)
    ],

    'RMSE': [

        np.sqrt(mean_squared_error(actual, ets_forecast)),

        np.sqrt(mean_squared_error(actual, arima_forecast)),

        np.sqrt(mean_squared_error(actual, rf_predictions))
    ]
})

st.dataframe(results)


# ============================================================
# FORECAST TABLE
# ============================================================

st.subheader("Forecast Table")

forecast_table = pd.DataFrame({

    'Actual': actual.values,

    'ETS': ets_forecast.values,

    'ARIMA': arima_forecast.values,

    'RandomForest': rf_predictions

}, index=test.index)

st.dataframe(forecast_table)


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

st.subheader("ARIMA Residual Diagnostics")

residuals = arima_model.resid


# Residual plot
fig, ax = plt.subplots(figsize=(14,4))

ax.plot(residuals)

ax.set_title("Residuals")

st.pyplot(fig)


# Histogram
fig, ax = plt.subplots(figsize=(8,4))

ax.hist(residuals, bins=25)

ax.set_title("Residual Distribution")

st.pyplot(fig)


# ACF
fig, ax = plt.subplots(figsize=(10,4))

plot_acf(residuals, ax=ax)

st.pyplot(fig)


# PACF
fig, ax = plt.subplots(figsize=(10,4))

plot_pacf(residuals, ax=ax)

st.pyplot(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("Random Forest Feature Importance")

importance_df = pd.DataFrame({

    'Feature': X_train.columns,

    'Importance': rf_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

st.dataframe(importance_df)


fig, ax = plt.subplots(figsize=(10,5))

ax.bar(
    importance_df['Feature'],
    importance_df['Importance']
)

ax.set_title("Feature Importance")

st.pyplot(fig)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.subheader("Download Results")

csv = forecast_table.to_csv().encode('utf-8')

st.download_button(
    label="Download Forecast Results",
    data=csv,
    file_name='forecast_results.csv',
    mime='text/csv'
)
