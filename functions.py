import math
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sksurv.nonparametric import kaplan_meier_estimator

import optuna
from sklearn.metrics import mean_absolute_error

def split_time_series_data(df, train_ratio=0.7, val_ratio=0.15):
    """
    Splits a DataFrame into Train, Validation, and Test sets 
    chronologically.
    """    
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]
    
    print(f"Total Rows: {n}")
    print(f"Train Size: {len(train_df)} ({train_ratio*100:.1f}%)")
    print(f"Val Size:   {len(val_df)} ({val_ratio*100:.1f}%)")
    print(f"Test Size:  {len(test_df)} ({(1 - (train_ratio + val_ratio))*100:.1f}%)")
    
    return train_df, val_df, test_df

def plot_stock_trends(df, column='Close', window=None, title='Stock Closing Prices Over Time'):
    """
    Plots historical trends for specified stock features across multiple tickers.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))

    if window:
        sns.lineplot(data=df[column].rolling(window=window).mean(), dashes=False)
        plt.title('Close Price with 30-Day Moving Average', fontsize=16)
    else:
        sns.lineplot(data=df[column], dashes=False)
        plt.title(title, fontsize=16)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(title='Tickers', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_decomposition_grid(df, tickers, period=7):
    """
    Creates a grid of Trend and Seasonal components for multiple tickers.
    2 plots per row: [Trend | Seasonal]
    """
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 2, figsize=(16, 4 * n_tickers))
    plt.subplots_adjust(hspace=0.4)

    for i, ticker in enumerate(tickers):
        series = df[('Close', ticker)].dropna()
        series.index = pd.to_datetime(series.index)
        series = series.asfreq('B').ffill() 

        decomposition = seasonal_decompose(series, model='additive', period=period)

        axes[i, 0].plot(decomposition.trend, color='blue')
        axes[i, 0].set_title(f'{ticker}: Trend Component', fontsize=12)
        axes[i, 0].tick_params(axis='x', rotation=45)

        axes[i, 1].plot(decomposition.seasonal.tail(30), color='green')
        axes[i, 1].set_title(f'{ticker}: Seasonal Component ({period} days)', fontsize=12)
        axes[i, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def plot_acf_grid(df, tickers, lags=40, column='High'):
    """
    Creates a grid of ACF plots for multiple tickers.
    2 plots per row.
    """
    n_tickers = len(tickers)
    n_cols = 2
    n_rows = math.ceil(n_tickers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        series = df[column][ticker].dropna()

        plot_acf(series, lags=lags, ax=axes[i], title=f'ACF: {ticker} {column}')
        
        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('Autocorrelation')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def run_stationarity_tests(df, tickers, column='High'):
    """
    Performs the Augmented Dickey-Fuller test on a list of tickers
    to determine stationarity.
    """
    for ticker in tickers:
        series = df[column][ticker].dropna()
        
        result = adfuller(series)
        
        print(f'ADF Test Results for {ticker} High')
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value:.4f}')

        if result[1] <= 0.05:
            print("\nThe series is Stationary.\n")
        else:
            print("\nThe series is Non-Stationary.\n")

def plot_differencing_comparison(df, tickers, column='High'):
    """
    Plots Original vs Differenced data in a grid.
    Uses dual Y-axes to handle the scale difference.
    """
    n_tickers = len(tickers)
    n_cols = 2
    n_rows = math.ceil(n_tickers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df[('High_Diff', ticker)] = df[('High', ticker)].diff()
        
        original_high = df[('High', ticker)]
        differenced_high = df[('High_Diff', ticker)]

        ax.plot(original_high, label=f'{ticker} Original High', color='blue', alpha=0.7)
        ax.plot(differenced_high, label=f'{ticker} Differenced High', linestyle='--', color='green')

        ax.legend()
        ax.set_title(f'Original vs Differenced High Price for {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price / Price Change')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def clean_and_verify_stationarity(df, tickers):
    """
    Cleans ticker data by calculating High_Diff, dropping NaNs, 
    and verifying stationarity via the ADF test.
    """
    cleaned_dfs = {}

    for ticker in tickers:
        print(f"TICKER: {ticker}")
        temp_df = pd.DataFrame({
            'Original_High': df['High'][ticker],
            'High_Diff': df['High'][ticker].diff(),
            'Volume': df['Volume'][ticker]
        }).copy()

        temp_df.dropna(subset=['High_Diff'], inplace=True)
        cleaned_dfs[ticker] = temp_df

        result = adfuller(temp_df['High_Diff'])
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        
        if result[1] <= 0.05:
            print("Result: Stationary (Ready for modeling)\n")
        else:
            print("Result: Non-Stationary (Needs further differencing)\n")

    return cleaned_dfs

def plot_pacf_grid(cleaned_dfs, tickers, lags=20):
    """
    Creates a grid of PACF plots for a dictionary of cleaned DataFrames.
    Arranged in 2 plots per row for efficient comparison.
    """
    n_tickers = len(tickers)
    n_cols = 2
    n_rows = math.ceil(n_tickers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        series = cleaned_dfs[ticker]['High_Diff'].dropna()
        plot_pacf(series, lags=lags, ax=axes[i], method='ywm')
        
        axes[i].set_title(f'PACF: {ticker} High_Diff', fontsize=14)
        axes[i].set_xlabel('Number of Lags (Days)', fontsize=10)
        axes[i].set_ylabel('Correlation Strength', fontsize=10)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def create_master_dataset(cleaned_dfs):
    """
    Applies technical indicators and lag features to each ticker,
    then concatenates them into a single master training DataFrame.
    """
    all_data = []

    for ticker, df in cleaned_dfs.items():
        df = df.copy()
        df['Ticker'] = ticker

        df['Vol_7d'] = df['High_Diff'].rolling(window=7).std()
        df['SMA_20'] = df['Original_High'].rolling(window=20).mean()
        df['Dist_SMA_20'] = (df['Original_High'] / df['SMA_20']) - 1
        
        delta = df['High_Diff']
        up = delta.clip(lower=0).rolling(window=14).mean()
        down = -delta.clip(upper=0).rolling(window=14).mean()
        rs = up / down.replace(0, np.nan) 
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        df['Vol_Mean_20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Std_20'] = df['Volume'].rolling(window=20).std()
        df['Vol_ZScore'] = (df['Volume'] - df['Vol_Mean_20']) / df['Vol_Std_20']

        ema_12 = df['Original_High'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Original_High'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26

        df['Upper_BB'] = df['SMA_20'] + (2 * df['Vol_7d'])
        df['Lower_BB'] = df['SMA_20'] - (2 * df['Vol_7d'])
        df['BB_Percent'] = (df['Original_High'] - df['Lower_BB']) / (df['Upper_BB'] - df['Lower_BB'])

        for lag in range(1, 8):
            df[f'lag_{lag}'] = df['High_Diff'].shift(lag)
        
        df['Target'] = df['High_Diff'].shift(-1)
        
        all_data.append(df)

    master_df = pd.concat(all_data)
    master_df.dropna(inplace=True)

    daily_market_avg = master_df.groupby(level=0)['High_Diff'].transform('mean')
    master_df['Relative_Performance'] = master_df['High_Diff'] - daily_market_avg

    return master_df

def prepare_feature_matrices(master_df):
    """
    Separates the master dataframe into feature matrix (X) and target vector (y).
    Groups features by type for easier preprocessing.
    """
    lag_features = [f'lag_{i}' for i in range(1, 8)]
    tech_indicators = [
        'Vol_7d', 'Dist_SMA_20', 'RSI', 'Vol_ZScore', 
        'MACD', 'BB_Percent', 'Relative_Performance'
    ]
    categorical_features = ['Ticker']

    all_features = categorical_features + lag_features + tech_indicators

    X = master_df[all_features]
    y = master_df['Target']
    
    return X, y

def plot_model_comparison(ranking_series, title='Overall Model Performance'):
    """
    Creates a horizontal bar chart to compare MAE across different models.
    Automatically labels each bar with its precise value.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    ax = ranking_series.plot(kind='barh')

    plt.title(title, fontsize=15)
    plt.xlabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.ylabel('Model', fontsize=12)

    for i, v in enumerate(ranking_series):
        ax.text(v + (ranking_series.max() * 0.02), i, f"{v:.4f}", 
                color='black', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_ticker_performance_comparison(comparison_df, ranking_avg, model_order, title='Model Performance by Ticker'):
    """
    Creates a grouped bar chart showing MAE for each ticker, 
    ordered by the overall average model performance.
    """

    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        data=comparison_df, 
        x='Model', 
        y='MAE', 
        hue='Ticker', 
        order=model_order
    )

    plt.title(f'{title} (Ordered by Average MAE)', fontsize=15)
    plt.ylabel('MAE (Error Amount)', fontsize=12)
    plt.xlabel('Model', fontsize=12)

    plt.legend(title='Tickers', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def get_best_params_for_model(study, model_name):
    """
    Filters an Optuna study to find the best performing trial for a specific model type.
    """
    model_trials = [
        t for t in study.trials 
        if t.params.get("model") == model_name and t.state == optuna.trial.TrialState.COMPLETE
    ]
    
    if not model_trials:
        return f"No trials found for {model_name}"
    
    best_model_trial = min(model_trials, key=lambda t: t.value)
    
    return {
        "params": best_model_trial.params,
        "mae": best_model_trial.value
    }

def clean_params(params_dict, prefix):
    """
    Cleans model parameter names for later use in model fitting.
    """
    new_params = {}

    for key, value in params_dict.items():
        if key == 'model':
            continue

        if key.startswith(prefix) or '_' not in key:
            clean_key = key.replace(prefix, '')
            new_params[clean_key] = value

    return new_params

def evaluate_ensemble(y_true, y_pred, title='Ensemble Voting Regressor'):
    """
    Calculates MAE and plots a comparison between actual and predicted 
    stock price movements for the first 100 observations.
    """
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{title} MAE: {mae:.6f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values[:100], label='Actual', alpha=0.7, color='green')
    plt.plot(y_pred[:100], label='Predicted', alpha=0.7, color='blue', linestyle='--')

    plt.title(f'Actual vs Predicted Stock Movement ({title})', fontsize=14)
    plt.xlabel('Time Steps (Validation Set)', fontsize=12)
    plt.ylabel('Price Change (High_Diff)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_kaplan_meier(df, duration_col="Duration", event_col="Event", threshold_label="5%"):
    """
    Computes and plots the Kaplan-Meier survival curve with confidence intervals.
    
    In this context, 'Survival' means the stock has NOT yet hit the target gain.
    """
    time, prob_surv, conf_int = kaplan_meier_estimator(
        df[event_col].astype(bool), 
        df[duration_col], 
        conf_type="log-log"
    )

    plt.figure(figsize=(8,6))

    plt.step(time, prob_surv, where="post", label=f"Global Survival ({threshold_label} Gain)")

    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")

    plt.ylim(0, 1)
    plt.xlim(0, time.max())
    plt.xlabel(f"Days to {threshold_label} Gain")
    plt.ylabel(f"Probability of NOT Hitting {threshold_label}")
    plt.title(f"Kaplan-Meier Estimate: Probability of Reaching {threshold_label} Gain Over Time")
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return time, prob_surv

def plot_ticker_survival_comparison(df, duration_col="Duration", event_col="Event"):
    """
    Plots independent Kaplan-Meier survival curves for each unique ticker in the dataset.
    Helps visualize which stocks reach the target threshold fastest.
    """
    plt.figure(figsize=(8,6))
    
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        
        time, prob_surv = kaplan_meier_estimator(
            ticker_data[event_col].astype(bool), 
            ticker_data[duration_col]
        )

        plt.step(time, prob_surv, where="post", label=f"Ticker: {ticker}")

    plt.ylim(0, 1)
    plt.xlim(0, df[duration_col].max())
    plt.xlabel("Days to Reach Target Gain", fontsize=12)
    plt.ylabel("Probability of NOT Hitting Target", fontsize=12)
    plt.title("Independent Survival Analysis per Stock", fontsize=15)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Tickers", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_rsf_predictions(model, X_samples, title="Predicted Time to 5% Gain"):
    """
    Plots the predicted survival functions from a Random Survival Forest
    for a given set of input samples.
    """
    surv_funcs = model.predict_survival_function(X_samples)

    plt.figure(figsize=(10, 6))

    for i, s in enumerate(surv_funcs):
        plt.step(s.x, s.y, where="post", label=f"Sample {i}")

    plt.ylabel("Survival Probability (Probability of No Hit)", fontsize=12)
    plt.xlabel("Days into the Future", fontsize=12)
    plt.title(f"{title} (Model Output)", fontsize=14)
    
    plt.legend(title="Test Samples", loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    
    plt.show()

def prep_ranking_data(df):
    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    cols_to_rel = ['RSI', 'Vol_ZScore', 'Dist_SMA_20', 'MACD']
    for col in cols_to_rel:
        df[f'{col}_day_mean'] = df.groupby('Date')[col].transform('mean')
        df[f'{col}_rel'] = df[col] - df[f'{col}_day_mean']
        
    df['relevance'] = df.groupby('Date')['High_Diff'].rank(method='first').astype(int) - 1
    return df

def pairwise_accuracy(group):    
    pairs = list(combinations(group.index, 2))
    correct = 0
    total = 0
    
    for i, j in pairs:
        actual_order = group.loc[i, 'relevance'] > group.loc[j, 'relevance']
        pred_order = group.loc[i, 'pred_score'] > group.loc[j, 'pred_score']
        
        if actual_order == pred_order:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0