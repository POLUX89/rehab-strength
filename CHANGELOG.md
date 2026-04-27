# Changelog
## [2.3.3] -2026/04/26
- Regression Learned models: OLS, Ridge, Lasso, Enet
- Learning curve
- Shap interpretation
- Unsupervised Models: PCA, T-SNE, K-Means

## [2.3.2] -2026/03/15
- Ramsey reset Test on OLS (Linearity)
- Durbin Watson Test (Autocorrelation)
- OLS with polynomial
- Stress_sleep feature gathered and added to the model
- Models included: Elastic Net, Ridge, Lasso, Decision Tree Regressor

## [2.3.1] -2026/02/21
Stats Tab
- Skewness & Kurtosis added
- Outliers values shown
Models Tab
- Model diagnostic (vih, cook`s distance, leverage)

## [2.3.0] -2026/02/04
-Toggle button to change between OLS and Logit
- OLS linear regression training below 150 samples
- OLS frozen at 150 samples
- OLS for deployment after 150 samples
- Deployment and training params
- OLS analysis (metrics, residuals)
- Correlation matrix
- ECDF and complementary ECDF on stats tab
- Hypothesis test for normality improved
- Learning curve in training phase for OLS model
- Stat Tab: Spearman correlation fully deployed

## [2.2.0] - 2026/01/28
-Stats tab
-Location estimates metrics
-CV metric
-Histogram, boxplots & lineplots for features
-Shapiro Wilk test for normality of distributions
-Outliers detection IQR & z-score modified
-Hypothesis testinf using ttest for normal distributions and mwu/correlation for non normal distributions

## [2.1.1] -2026/01/28
-Tag HRV metric on home tab as Excellent, Good or Bad
-Time exercised metric on home tab with a goal of 4 hours
-Recovery plot with nap on home tab
-Home tab recovery charts with insights
-Error handling with naps metrics

## [2.1.0] - 2026/01/20
-Naps added to sleep tab
-Home tab with nap metrics (average, nap days & nap frequency)
-Naps classified based on the duration & the hour it was taken
-Sigmoid recovery function modified by naps
-Home tab with naps summary (sigmoid modified, delta & status)

## [2.0.4] -2026/01/18
-Fixed data governance for better accuracy
-Visuals improved

## [2.0.3] -2026/01/17
- Sliders with moving average up to 11 days added for plot recovery and sleep charts in home tab
-Dark mode always ON
-Standardize y-axis scaling for consistent interpretation


## [2.0.2] - 2026/01/16
- Data freshness badge (workouts / sleep / recovery)
- Home tab data status indicator (slightly delayed / fresh)

## [2.0.1]
- Data integrity added as a visible metric in Home tab

## [2.0.0]
- Fixed major bugs
- Collapsed file upload panel
- Added data integrity checks
- Added pd.merge logic in import_sleep_data.py
- Refactored app architecture

## [1.5.0]
- Added 5 tabs: Home, Workouts, Sleep, Recovery, Correlations
- Introduced sigmoid recovery score (experimental)

## [1.0.0]
- Initial app creatioN

