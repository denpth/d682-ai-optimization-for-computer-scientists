# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# Load the dataset and describe the data
df = pd.read_csv('dataset.csv')
print("Shape:")
print(df.shape)
print("Head:")
print(df.head())
print("Info:")
print(df.info())
print("Describe:")
print(df.describe())
print("Null values:")
print(df.isnull().sum())

# Drop day of the week and is weekend because these are not related to air quality data. Month is dropped because it is static and will have no affect on analysis.
df.drop(['month'], axis=1, inplace=True)
df.drop(['dayOfWeek', 'isWeekend'], axis=1, inplace=True)
print("Dropped month, day of week and isweekend:")
df.head()

# Convert datetimeEpoch to datetime format to visualize actual data
df['datetime'] = pd.to_datetime(df['datetimeEpoch'], unit='s')
df.set_index('datetime', inplace=True)
df.drop('datetimeEpoch', axis=1, inplace=True)
print("Converted Epochs to datetime format:")
print(df.head())

# Visualize data focused on key variables related to health risk
print("Visualizing key health-related data:")
health_vars = ['temp', 'feelslike', 'pm2.5', 'no2', 'co2', 'humidity', 'uvindex', 'healthRiskScore']
sns.pairplot(df[health_vars], diag_kind='hist')
plt.suptitle('Health Risk Factors Pairplot', y=1.02)
plt.show()

# Visualize data with multiple focused visualizations
print("Visualizing data with multiple plots:")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Temperature vs Health Risk Score
sns.scatterplot(data=df, x='temp', y='healthRiskScore', ax=axes[0,0])
axes[0,0].set_title('Temperature vs Health Risk Score')

# 2. Air Quality vs Health Risk Score
sns.scatterplot(data=df, x='pm2.5', y='healthRiskScore', ax=axes[0,1])
axes[0,1].set_title('PM2.5 vs Health Risk Score')

# 3. Distribution of Health Risk Scores
sns.histplot(data=df, x='healthRiskScore', bins=30, ax=axes[1,0])
axes[1,0].set_title('Distribution of Health Risk Scores')

# 4. Correlation heatmap of key variables
key_vars = ['temp', 'feelslike', 'pm2.5', 'no2', 'co2', 'humidity', 'uvindex', 'healthRiskScore']
correlation_matrix = df[key_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()


# Visualize data over time
print("Visualizing time series data:")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot temperature over time
axes[0].plot(df.index, df['temp'], label='Temperature', color='red', alpha=0.7)
axes[0].plot(df.index, df['feelslike'], label='Feels Like', color='orange', alpha=0.7)
axes[0].set_title('Temperature Over Time')
axes[0].set_ylabel('Temperature (°F)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot air quality over time
axes[1].plot(df.index, df['pm2.5'], label='PM2.5', color='brown', alpha=0.7)
axes[1].plot(df.index, df['no2'], label='NO2', color='purple', alpha=0.7)
axes[1].set_title('Air Quality Over Time')
axes[1].set_ylabel('Concentration')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot health risk score over time
axes[2].plot(df.index, df['healthRiskScore'], label='Health Risk Score', color='darkred', linewidth=2)
axes[2].set_title('Health Risk Score Over Time')
axes[2].set_ylabel('Health Risk Score')
axes[2].set_xlabel('Date')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize data with simple correlation analysis
print("Correlation with Health Risk Score:")

# Calculate correlations with health risk score
correlations = df.corr()['healthRiskScore'].sort_values(ascending=False)
print(correlations)

# Visualize top correlations
top_correlations = correlations.drop('healthRiskScore').head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_correlations.values, y=top_correlations.index)
plt.title('Top 10 Features Correlated with Health Risk Score')
plt.xlabel('Correlation Coefficient')
plt.show()

# Drop sunrise and sunset data to ensure it does not obscure data for model
df.drop(['sunriseEpoch', 'sunsetEpoch'], axis=1, inplace=True)
print("Dropped sunrise and sunset data:")
print(df.head())

# Split data into features and target
X = df.drop('healthRiskScore', axis=1)
y = df['healthRiskScore']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Fit on TRAIN
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Holdout evaluation on Test
rmse_test = root_mean_squared_error(y_test, y_pred)
r2_test   = r2_score(y_test, y_pred)

print("Holdout RMSE:", rmse_test)
print("Holdout R²:",  r2_test)

# Cross-validation on TRAIN ONLY
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}

scores = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

rmse_cv = -scores["test_rmse"]
r2_cv   =  scores["test_r2"]

print("CV RMSE (mean ± std): {:.4f} ± {:.4f}".format(rmse_cv.mean(), rmse_cv.std()))
print("CV R²   (mean ± std): {:.4f} ± {:.4f}".format(r2_cv.mean(), r2_cv.std()))