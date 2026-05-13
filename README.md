# DS.v2.5.3.3.5

# Stock analysis

## Introduction

The project analyzes the historical stock performance of five major technology companies (AAPL, AMZN, GOOGL, META, MSFT) to forecast future movements, evaluate time-to-gain probabilities, and rank relative daily performance.

### Data overview

- Timeframe: The dataset contains 4085 days of data, representing over 16 years of market activity.

- General Trend: All five technology stocks display a clear upward trajectory, with notable market dips primarily occurring around 2020.

- Splitting Strategy: To prevent data leakage and ensure the models cannot "see" the future, the data was split chronologically - the earliest data for training, the middle period for validation, and the most recent data for testing.

### Variables

### Variables

Time Series & Base Features: 
- 7-day rolling standard deviation,
- 20-day Simple Moving Average,
- normalized distance from the 20-day SMA,
- Moving Average Convergence Divergence,
- 14-day Relative Strength Index,
- Bollinger Bands,
- 20-day rolling volume mean and standard deviation,
- volume Z-score, 
- historical price change lags.

Survival Analysis Additions: 
- Current daily percentage return,
- days until a >5% gain is reached,
- binary censoring indicator.

Ranking Model Additions: 
- Cross-sectional normalized features (daily mean subtracted for RSI, Vol_ZScore, Dist_SMA_20, and MACD), 
- overall market return, 
- overall market volatility,
- the target variable is an integer relevance score.

### Data limitations

- Missing Values: META contains missing values because its stock data begins on May 18, 2012, whereas the starting date for the other four stocks is January 4, 2010.

- Non-Stationarity: The raw stock prices show high autocorrelation and failed the Augmented Dickey-Fuller (ADF) test. Differencing the high prices was required to remove the trend and achieve stationarity for modeling.

- Lack of Seasonality: The raw stock data possesses no reliable seasonality; minor patterns are considered mathematical noise relative to the scale of the overarching trend.

### Goal of the analysis

Build three models that can:

- forecast price changes

- predict the probability and time needed for a stock price to rise 5% or more per day

- rank the daily performance of the stocks

### Key findings

Stock Profiles:

- META: Displayed the fastest growth and highest prices, but also the most intense, unpredictable swings (yielding the highest MAE). Its survival curve drops the fastest, indicating it hits 5% gains very quickly.

- MSFT: Exhibited steady growth with the fewest explosive jumps, reflected by the highest survival curve.

- AMZN: Showed high volatility and reached the 5% target quickly, similar to META.

- GOOGL: Had the lowest MAE across all models, indicating its price fluctuations align best with the engineered features.

- AAPL: Held the lowest price until 2016 but surpassed GOOGL after 2020.

Feature Efficacy: A 7-day lag is the optimal middle ground for capturing weekly patterns across all stocks. The strong performance of linear models indicates a linear relationship between the engineered features and the targets.

Survival Insights: The survival model accurately ranks time-to-gain 71% of the time (C-index: 0.7075). It identified that if a stock has not hit a 5% gain by day 500, it is certain to do so immediately after.

Ranking Accuracy: The ranking model correctly identifies the better performing stock in a pairwise comparison 67.92% of the time on validation data, maintaining a consistent 67.2% on testing data.

### Models used

- Time Series Forecasting: CatBoost (best performing base model), Ridge Regression, XGBoost, LightGBM, and RandomForest. Stacking Ensemble used the Ridge model as the estimator to determine when to rely on the predictions of the other models.

- Survival Analysis: Random Survival Forest.

- Ranking: LGBMRanker, evaluated using Normalized Discounted Cumulative Gain (NDCG).

### Conclusions

- GOOGL is the most consistent stock for price forecasting (lowest MAE), whereas META’s high volatility makes it the fastest to reach a 5% gain threshold despite its lower daily predictability.

- A stacking ensemble that utilizes Ridge regression as the estimator for gradient-boosted models provides the most accurate forecasts for intense market highs and lows.

- Survival model reached 71% time-to-gain accuracy

- Ranking model achieved 68% pairwise accuracy

