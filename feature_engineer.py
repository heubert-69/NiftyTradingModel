import numpy as np
import pandas as pd

def engineer_features_and_labels(
    df: pd.DataFrame,
    horizon: int = 1,
    rolling_window: int = 20
):
    """
    Engineer features + generate labels for trading strategy + performance metrics.
    Expects df with columns: ['datetime','open','high','low','close','volume','oi'].
    horizon = how many candles ahead to calculate return.
    rolling_window = window size for rolling Sharpe.
    
    Returns:
        X (pd.DataFrame): Feature matrix aligned with model expectations.
        y (pd.Series): Labels for supervised learning.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ==============================
    # EMA features
    # ==============================
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    
    df['ema_dist'] = df['ema_9'] - df['ema_15']
    df['ema9_slope'] = df['ema_9'].diff(3)
    df['ema15_slope'] = df['ema_15'].diff(3)
    df['ema9_angle'] = np.degrees(np.arctan(df['ema9_slope']))
    df['ema15_angle'] = np.degrees(np.arctan(df['ema15_slope']))
    df['trend_up'] = (df['ema_9'] > df['ema_15']).astype(int)

    # ==============================
    # Candlestick features
    # ==============================
    df['candle_body'] = df['close'] - df['open']
    df['candle_range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - df[['close','open']].max(axis=1)
    df['lower_shadow'] = df[['close','open']].min(axis=1) - df['low']
    df['body_ratio'] = (df['candle_body'].abs() / df['candle_range']).replace([np.inf, -np.inf], 0)

    df['is_pin_bar'] = ((df['upper_shadow'] > 2 * df['candle_body'].abs()) |
                        (df['lower_shadow'] > 2 * df['candle_body'].abs())).astype(int)
    df['is_full_body'] = (df['body_ratio'] > 0.7).astype(int)
    df['is_big_bar'] = (df['candle_range'] > df['candle_range'].rolling(20).mean()).astype(int)

    # ==============================
    # EMA touches
    # ==============================
    df['touch_ema9'] = ((df['low'] <= df['ema_9']) & (df['high'] >= df['ema_9'])).astype(int)
    df['touch_ema15'] = ((df['low'] <= df['ema_15']) & (df['high'] >= df['ema_15'])).astype(int)

    # ==============================
    # Flat market filter + metrics
    # ==============================
    df['volatility'] = df['close'].pct_change().rolling(10).std()
    threshold = df['volatility'].quantile(0.2) if not df['volatility'].isna().all() else 0
    df['is_flat'] = (df['volatility'] < threshold).astype(int)

    df['returns'] = df['close'].pct_change()
    df['cum_returns'] = (1 + df['returns'].fillna(0)).cumprod() - 1
    df['rolling_sharpe'] = (
        df['returns'].rolling(rolling_window).mean() /
        df['returns'].rolling(rolling_window).std()
    )
    df['drawdown'] = df['cum_returns'] - df['cum_returns'].cummax()

    # ==============================
    # Labels (future return classification)
    # ==============================
    df['future_return'] = (df['close'].shift(-horizon) - df['close']) / df['close']

    def label_trade(r):
        if r >= 0.0015:   # +0.15%
            return 0  # Buy+
        elif r >= 0.001:  # +0.10%
            return 1  # Buy
        elif r <= -0.0015: # -0.15%
            return 4  # Sell+
        elif r <= -0.001:  # -0.10%
            return 3  # Sell
        else:
            return 2  # Do nothing

    df['target'] = df['future_return'].apply(lambda x: label_trade(x) if pd.notnull(x) else np.nan)

    # ==============================
    # Final features (align with model expectations)
    # ==============================
    feature_cols = [
        'datetime','open','high','low','close','volume','oi',
        'ema_dist','ema9_angle','ema15_angle','trend_up',
        'candle_body','body_ratio','is_pin_bar','is_full_body','is_big_bar',
        'touch_ema9','touch_ema15','is_flat',
        'returns','cum_returns','rolling_sharpe','drawdown'
    ]

    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols]
    y = df['target']

    return X, y
def preprocess_for_sklearn(X, y=None, apply_smote=False):
    X_processed = X.copy()

    # Handle data types
    for col in X_processed.columns:
        if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
            X_processed[col] = X_processed[col].astype('int64') // 10**9
        elif X_processed[col].dtype == 'object':
            try:
                datetime_col = pd.to_datetime(X_processed[col])
                X_processed[col] = datetime_col.astype('int64') // 10**9
            except (ValueError, TypeError):
                X_processed[col] = pd.factorize(X_processed[col])[0]

    # Fill missing values
    X_processed = X_processed.fillna(X_processed.mean())

    # Optionally apply SMOTE
    if apply_smote and y is not None:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_processed, y)
        return X_res, y_res

    return X_processed, y

