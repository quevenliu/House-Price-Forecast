import joblib
import lightgbm as lgb
from pipeline import pipeline
import pandas as pd
from tqdm import tqdm

# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Preprocess X_train and y_train
X_train_processed = pipeline.fit_transform(X_train)
y_train_processed = y_train['單價元平方公尺'].values

# pickle dump pipeline
joblib.dump(pipeline, 'lightgbm-pipeline.pkl')

# Define Dataset for LightGBM
dtrain = lgb.Dataset(X_train_processed, label=y_train_processed)

# Set up parameter grid for tuning
param_grid = {
    'num_leaves': [64, 128, 256],
    'learning_rate': [0.3, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'lambda_l1': [0, 0.5, 1]
}

# Fixed parameters
fixed_params = {
    'objective': 'regression',  # Set the objective to regression
    'metric': 'rmse',           # Use RMSE as the evaluation metric
    'boosting_type': 'gbdt'
}

# Cross-validation parameters
nfold = 5

# Store results
best_score = float("Inf")
best_params = None

# Define function to run cross-validation for a single set of parameters


def evaluate_params(params):
    # Perform cross-validation with the provided parameters
    cv_results = lgb.cv(
        params=params,
        train_set=dtrain,
        nfold=nfold,
        seed=42,
        stratified=False,
        metrics='rmse',
    )

    # Get the best score for the current parameter set
    mean_rmse = min(cv_results['valid rmse-mean'])  # Best RMSE score
    # Index of best score
    best_iteration = cv_results['valid rmse-mean'].index(mean_rmse)

    return mean_rmse, best_iteration, params


# Generate all parameter combinations
all_params = [
    {**fixed_params, 'num_leaves': num_leaves, 'learning_rate': learning_rate, 'subsample': subsample,
     'colsample_bytree': colsample_bytree, 'lambda_l1': reg_alpha}
    for num_leaves in param_grid['num_leaves']
    for learning_rate in param_grid['learning_rate']
    for subsample in param_grid['subsample']
    for colsample_bytree in param_grid['colsample_bytree']
    for reg_alpha in param_grid['lambda_l1']
]

# Initialize best score and parameters
best_score = float("Inf")
best_params = None

# Sequential evaluation of all parameter combinations
for params in tqdm(all_params):
    mean_rmse, best_iteration, params = evaluate_params(params)

    # Update best score and parameters if the current score is better
    if mean_rmse < best_score:
        best_score = mean_rmse
        best_params = params

# Print best parameters and score
print("Best Parameters:", best_params)
print("Best RMSE:", best_score)

# Train final model with best parameters
final_model = lgb.LGBMRegressor(
    **best_params
)
final_model.fit(X_train_processed, y_train_processed)

# Save the model
joblib.dump(final_model, 'lgb-model.pkl')

# Predict on X_test.csv
X_test = pd.read_csv('data/X_test.csv')
X_test_processed = pipeline.transform(X_test)
y_test = final_model.predict(X_test_processed)
y_test_df = pd.DataFrame(y_test, columns=['單價元平方公尺'])
y_test_df['Id'] = X_test['Id']
y_test_df = y_test_df[['Id', '單價元平方公尺']]
# Print to float with one decimal place
y_test_df['單價元平方公尺'] = y_test_df['單價元平方公尺'].round(1)
y_test_df.to_csv('data/y_test.csv', index=False)
