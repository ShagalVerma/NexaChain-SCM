"""
train_model.py
Trains per-product Random Forest demand forecasting models.
Generates forecasts for Jan – Jun 2025 and saves everything to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
demand_df = pd.read_csv('data/demand_history.csv')
demand_df['date'] = pd.to_datetime(demand_df['date'])

# Aggregate demand: sum all regions per product per month
agg = (demand_df
       .groupby(['product_id', 'product_name', 'category', 'year_month', 'date'])
       ['actual_demand'].sum()
       .reset_index()
       .sort_values(['product_id', 'date']))

# Also aggregate manual forecast for comparison
manual_agg = (demand_df
              .groupby(['product_id', 'year_month'])
              ['manual_forecast'].sum()
              .reset_index())

agg = agg.merge(manual_agg, on=['product_id', 'year_month'])


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
def create_features(df):
    df = df.copy().reset_index(drop=True)
    df['month_num'] = df['date'].dt.month
    df['year']      = df['date'].dt.year
    df['quarter']   = df['date'].dt.quarter
    df['trend']     = np.arange(len(df))

    # Lag features
    df['lag_1']  = df['actual_demand'].shift(1)
    df['lag_2']  = df['actual_demand'].shift(2)
    df['lag_3']  = df['actual_demand'].shift(3)
    df['lag_6']  = df['actual_demand'].shift(6)
    df['lag_12'] = df['actual_demand'].shift(12)

    # Rolling statistics
    df['roll_mean_3']  = df['actual_demand'].shift(1).rolling(3).mean()
    df['roll_mean_6']  = df['actual_demand'].shift(1).rolling(6).mean()
    df['roll_std_3']   = df['actual_demand'].shift(1).rolling(3).std().fillna(0)
    df['roll_std_6']   = df['actual_demand'].shift(1).rolling(6).std().fillna(0)

    # Seasonality flags (India)
    df['is_festive']    = df['month_num'].isin([9, 10, 11]).astype(int)
    df['is_new_year']   = df['month_num'].isin([1, 12]).astype(int)
    df['is_low_season'] = df['month_num'].isin([4, 5]).astype(int)

    # Month sin/cos encoding (captures cyclical nature)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    return df

FEATURES = [
    'month_num', 'year', 'quarter', 'trend',
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'roll_mean_3', 'roll_mean_6', 'roll_std_3', 'roll_std_6',
    'is_festive', 'is_new_year', 'is_low_season',
    'month_sin', 'month_cos',
]


# ─────────────────────────────────────────────
# Train and forecast
# ─────────────────────────────────────────────
all_forecasts = []
all_metrics   = []

for pid in agg['product_id'].unique():
    prod_df   = agg[agg['product_id'] == pid].copy().reset_index(drop=True)
    prod_name = prod_df['product_name'].iloc[0]
    category  = prod_df['category'].iloc[0]

    feat_df = create_features(prod_df)
    feat_df = feat_df.dropna(subset=FEATURES)

    X = feat_df[FEATURES]
    y = feat_df['actual_demand']

    # Train / test  (last 6 months = test set)
    split      = max(len(X) - 6, int(len(X) * 0.8))
    X_train    = X.iloc[:split]
    X_test     = X.iloc[split:]
    y_train    = y.iloc[:split]
    y_test     = y.iloc[split:]

    # Main model: Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)

    # Metrics
    mae      = mean_absolute_error(y_test, y_pred_test)
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape     = np.mean(np.abs((y_test.values - y_pred_test) / y_test.values)) * 100
    accuracy = max(0, round(100 - mape, 2))

    # Manual forecast metrics (for comparison)
    manual_test  = feat_df['manual_forecast'].iloc[split:]
    manual_mape  = np.mean(np.abs((y_test.values - manual_test.values) / y_test.values)) * 100
    manual_acc   = max(0, round(100 - manual_mape, 2))

    all_metrics.append({
        'product_id':         pid,
        'product_name':       prod_name,
        'category':           category,
        'MAE':                round(mae, 1),
        'RMSE':               round(rmse, 1),
        'MAPE_pct':           round(mape, 2),
        'RF_Accuracy_pct':    accuracy,
        'Manual_Accuracy_pct': manual_acc,
        'Improvement_pct':    round(accuracy - manual_acc, 2),
    })

    # ── Historical predictions ──
    hist_preds = rf.predict(X)
    for i, row in feat_df.iterrows():
        std_val = row['roll_std_6'] if row['roll_std_6'] > 0 else row['actual_demand'] * 0.05
        pred    = max(0, int(hist_preds[list(feat_df.index).index(i)]))
        all_forecasts.append({
            'product_id':       pid,
            'product_name':     prod_name,
            'category':         category,
            'year_month':       row['year_month'],
            'date':             row['date'],
            'actual_demand':    int(row['actual_demand']),
            'manual_forecast':  int(row['manual_forecast']),
            'predicted_demand': pred,
            'lower_bound':      max(0, int(pred - 1.5 * std_val)),
            'upper_bound':      int(pred + 1.5 * std_val),
            'is_future':        False,
        })

    # ── Future forecast: Jan – Jun 2025 ──
    recent_vals = list(prod_df['actual_demand'].values)
    future_dates = pd.date_range(start='2025-01-01', periods=6, freq='MS')

    for j, fdate in enumerate(future_dates):
        trend_val  = len(feat_df) + j
        lag1       = recent_vals[-1]
        lag2       = recent_vals[-2]
        lag3       = recent_vals[-3]
        lag6       = recent_vals[-6]  if len(recent_vals) >= 6  else recent_vals[0]
        lag12      = recent_vals[-12] if len(recent_vals) >= 12 else recent_vals[0]
        roll3_mean = np.mean(recent_vals[-3:])
        roll6_mean = np.mean(recent_vals[-6:])
        roll3_std  = np.std(recent_vals[-3:])
        roll6_std  = np.std(recent_vals[-6:])
        m          = fdate.month

        future_row = pd.DataFrame([[
            m, fdate.year, (m - 1) // 3 + 1, trend_val,
            lag1, lag2, lag3, lag6, lag12,
            roll3_mean, roll6_mean, roll3_std, roll6_std,
            int(m in [9, 10, 11]),
            int(m in [1, 12]),
            int(m in [4, 5]),
            np.sin(2 * np.pi * m / 12),
            np.cos(2 * np.pi * m / 12),
        ]], columns=FEATURES)

        pred     = max(0, int(rf.predict(future_row)[0]))
        std_val  = roll6_std if roll6_std > 0 else pred * 0.08

        all_forecasts.append({
            'product_id':       pid,
            'product_name':     prod_name,
            'category':         category,
            'year_month':       fdate.strftime('%Y-%m'),
            'date':             fdate,
            'actual_demand':    None,
            'manual_forecast':  None,
            'predicted_demand': pred,
            'lower_bound':      max(0, int(pred - 1.5 * std_val)),
            'upper_bound':      int(pred + 1.5 * std_val),
            'is_future':        True,
        })

        recent_vals.append(pred)

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
forecasts_df = pd.DataFrame(all_forecasts)
metrics_df   = pd.DataFrame(all_metrics)

forecasts_df.to_csv('data/forecasts.csv', index=False)
metrics_df.to_csv('data/model_metrics.csv', index=False)

print("✅ Model training complete!")
print("\n📊 Model Performance (RF vs Manual Forecast):")
print(metrics_df[['product_name', 'RF_Accuracy_pct', 'Manual_Accuracy_pct', 'Improvement_pct']].to_string(index=False))
avg_acc = metrics_df['RF_Accuracy_pct'].mean()
avg_man = metrics_df['Manual_Accuracy_pct'].mean()
print(f"\n   Average RF Accuracy     : {avg_acc:.1f}%")
print(f"   Average Manual Accuracy : {avg_man:.1f}%")
print(f"   ML Improvement          : +{avg_acc - avg_man:.1f}%")
