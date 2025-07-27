# Import libraries
import pandas as pd
import numpy as np
# Visualization imports removed - only mathematical comparison needed
from sklearn.model_selection import cross_validate, train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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

# Data analysis - key health-related variables
print("Analyzing key health-related data:")
health_vars = ['temp', 'feelslike', 'pm2.5', 'no2', 'co2', 'humidity', 'uvindex', 'healthRiskScore']


# Data correlation analysis
print("Correlation with Health Risk Score:")

# Calculate correlations with health risk score
correlations = df.corr()['healthRiskScore'].sort_values(ascending=False)
print("Top 10 correlations with Health Risk Score:")
print(correlations.head(10))

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

# TASK 2 STARTS HERE

print("\n" + "="*80)
print("PART B: OPTIMIZATION, REGULARIZATION, AND ENSEMBLE TECHNIQUES")
print("="*80)

# Store baseline performance for comparison
baseline_rmse = rmse_test
baseline_r2 = r2_test
baseline_rmse_cv = rmse_cv.mean()
baseline_r2_cv = r2_cv.mean()

print(f"\nBaseline Performance:")
print(f"Test RMSE: {baseline_rmse:.4f}")
print(f"Test R²: {baseline_r2:.4f}")
print(f"CV RMSE: {baseline_rmse_cv:.4f}")
print(f"CV R²: {baseline_r2_cv:.4f}")

# ============================================================================
# STEP 1: FEATURE SELECTION
# ============================================================================
print("\n" + "-"*60)
print("STEP 1: FEATURE SELECTION")
print("-"*60)

# Apply SelectKBest feature selection
from sklearn.feature_selection import SelectKBest, f_regression

# Select top 15 features based on F-statistic
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} features:")
print(selected_features)

# Update our training data to use selected features
X_train = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)

# ============================================================================
# STEP 2: HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "-"*60)
print("STEP 2: HYPERPARAMETER TUNING")
print("-"*60)

# Define parameter distributions for RandomizedSearchCV
param_distributions = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
    'bootstrap': [True, False]
}

# Perform randomized search on selected features
rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Performing hyperparameter tuning on selected features...")
rf_random.fit(X_train, y_train)

print(f"Best parameters found: {rf_random.best_params_}")
print(f"Best CV score: {-rf_random.best_score_:.4f}")

# Store optimized parameters for later use
best_params = rf_random.best_params_

# ============================================================================
# STEP 3: RIDGE REGRESSION
# ============================================================================
print("\n" + "-"*60)
print("STEP 3: RIDGE REGRESSION")
print("-"*60)

# Scale the selected features for Ridge regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Ridge regression with cross-validation
from sklearn.linear_model import RidgeCV

ridge_alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=5, scoring='neg_root_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

print(f"Ridge Regression - Best alpha: {ridge_cv.alpha_}")

# Evaluate Ridge performance
ridge_pred = ridge_cv.predict(X_test_scaled)
ridge_rmse = root_mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge RMSE: {ridge_rmse:.4f}, R²: {ridge_r2:.4f}")

# ============================================================================
# STEP 4: REGULARIZED RANDOM FOREST PARAMETERS
# ============================================================================
print("\n" + "-"*60)
print("STEP 4: REGULARIZED RANDOM FOREST PARAMETERS")
print("-"*60)

# Apply regularization constraints to the optimized Random Forest
# Use the best parameters from hyperparameter tuning but add regularization
regularized_params = best_params.copy()
regularized_params.update({
    'max_depth': min(regularized_params.get('max_depth', 20), 15),  # Limit depth
    'min_samples_split': max(regularized_params.get('min_samples_split', 2), 10),  # Increase min samples
    'min_samples_leaf': max(regularized_params.get('min_samples_leaf', 1), 5),     # Increase min leaf samples
    'max_features': 'sqrt'  # Limit features per split
})

print(f"Regularized parameters: {regularized_params}")

# Create regularized Random Forest with optimized + regularized parameters
rf_regularized = RandomForestRegressor(**regularized_params, random_state=42, n_jobs=-1)
rf_regularized.fit(X_train, y_train)

# Evaluate regularized RF performance
rf_reg_pred = rf_regularized.predict(X_test)
rf_reg_rmse = root_mean_squared_error(y_test, rf_reg_pred)
rf_reg_r2 = r2_score(y_test, rf_reg_pred)
print(f"Regularized RF RMSE: {rf_reg_rmse:.4f}, R²: {rf_reg_r2:.4f}")

print("Training regularized Random Forest on selected features...")

# ============================================================================
# STEP 5: ENSEMBLE TECHNIQUE 1 - Voting Ensemble
# ============================================================================
print("\n" + "-"*60)
print("STEP 5: ENSEMBLE TECHNIQUE 1 - Voting Ensemble")
print("-"*60)

# Create individual models for the voting ensemble
rf_for_ensemble = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
gb_for_ensemble = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Create voting ensemble (using models that work well together)
voting_ensemble = VotingRegressor([
    ('rf', rf_for_ensemble),
    ('gb', gb_for_ensemble),
    ('ridge', ridge_cv)
])

print("Training voting ensemble...")
# Train on scaled features for consistency
voting_ensemble.fit(X_train_scaled, y_train)

# Evaluate voting ensemble performance
voting_pred = voting_ensemble.predict(X_test_scaled)
voting_rmse = root_mean_squared_error(y_test, voting_pred)
voting_r2 = r2_score(y_test, voting_pred)
print(f"Voting Ensemble RMSE: {voting_rmse:.4f}, R²: {voting_r2:.4f}")

# ============================================================================
# STEP 6: ENSEMBLE TECHNIQUE 2 - Gradient Boosting Ensemble
# ============================================================================
print("\n" + "-"*60)
print("STEP 6: ENSEMBLE TECHNIQUE 2 - Gradient Boosting Ensemble")
print("-"*60)

# XGBoost with selected features
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# Evaluate XGBoost performance
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = root_mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")

# Create final ensemble combining all techniques
print("\nCreating final ensemble combining all techniques...")
final_ensemble = VotingRegressor([
    ('rf_regularized', rf_regularized),
    ('voting_ensemble', voting_ensemble),
    ('xgb', xgb_model)
], weights=[0.3, 0.4, 0.3])

# Train final ensemble on scaled features
final_ensemble.fit(X_train_scaled, y_train)

# Perform final prediction
final_prediction = final_ensemble.predict(X_test_scaled)

