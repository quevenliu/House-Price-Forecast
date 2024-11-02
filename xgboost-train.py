import joblib
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Custom Transformer for Date Columns and Calculations


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Parse dates and create '屋齡' column
        X = X.copy()
        X['建築完成年月'] = pd.to_datetime(X['建築完成年月'])
        X['交易日期'] = pd.to_datetime(X['交易年'].astype(
            str) + '-' + X['交易月'].astype(str) + '-' + X['交易日'].astype(str))
        X['屋齡'] = (X['交易日期'] - X['建築完成年月']).dt.days
        # Drop unnecessary columns
        return X.drop(columns=['建築完成年月', '交易年', '交易月', '交易日', '交易日期'])

# Custom Transformer for Boolean Mapping


class BooleanMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column, mapping in self.mapping_dict.items():
            X[column] = X[column].map(mapping)
        return X


# Columns for each transformer
columns_to_drop = ['路名']
one_hot_columns = ['鄉鎮市區', '交易標的', '移轉層次項目',
                   '都市土地使用分區', '建物型態', '主要用途', '主要建材']
boolean_columns_mapping = {
    '建物現況格局-隔間': {'有': True, '無': False},
    '有無管理組織': {'有': True, '無': False}
}
columns_to_normalize = ['土地移轉總面積平方公尺', '建物移轉總面積平方公尺', '縱坐標', '橫坐標', '屋齡']

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('drop_columns', 'drop', columns_to_drop),
        ('boolean_mapping', BooleanMappingTransformer(
            mapping_dict=boolean_columns_mapping), list(boolean_columns_mapping.keys())),
        # Handle unknown categories in one-hot encoding
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), one_hot_columns),
        ('scaling', StandardScaler(), columns_to_normalize)
    ],
    remainder='passthrough'
)

# Full pipeline with date transformation
pipeline = Pipeline([
    ('date_transform', DateTransformer()),
    ('preprocessing', preprocessor)
])

# Preprocess X_train and y_train
X_train_processed = pipeline.fit_transform(X_train)
y_train_processed = y_train['單價元平方公尺'].values

# Define DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_processed, label=y_train_processed)

# Set up parameter grid for tuning

param_grid = {
    'max_depth': [3, 5, 7],               # Tree depth for base learners
    'eta': [0.3, 0.1, 0.01],            # Learning rate
    # Subsample ratio of training instances
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],  # Subsample ratio of columns
    'lambda': [1, 1.5, 2],                # L2 regularization term
    'alpha': [0, 0.5, 1]                  # L1 regularization term
}


# Fixed parameters
fixed_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Define cross-validation parameters
early_stopping_rounds = 10
nfold = 5

# Store results
best_score = float("Inf")
best_params = None

# Define function to run cross-validation for a single set of parameters


def evaluate_params(params):
    # Perform cross-validation with the provided parameters
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        nfold=nfold,
        early_stopping_rounds=early_stopping_rounds,
        seed=42,
        verbose_eval=False
    )

    # Get the best score and iteration for the current parameter set
    mean_rmse = cv_results['test-rmse-mean'].min()
    best_iteration = cv_results['test-rmse-mean'].idxmin()

    # Return the score, best iteration, and parameters
    return mean_rmse, best_iteration, params


# Generate all parameter combinations
all_params = [
    {**fixed_params, 'max_depth': max_depth, 'eta': eta, 'subsample': subsample,
     'colsample_bytree': colsample_bytree, 'lambda': reg_lambda, 'alpha': reg_alpha}
    for max_depth in param_grid['max_depth']
    for eta in param_grid['eta']
    for subsample in param_grid['subsample']
    for colsample_bytree in param_grid['colsample_bytree']
    for reg_lambda in param_grid['lambda']
    for reg_alpha in param_grid['alpha']
]

# Use ThreadPoolExecutor to run evaluations in parallel
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(evaluate_params, params)               : params for params in all_params}
    for future in as_completed(futures):
        mean_rmse, best_iteration, params = future.result()

        print("future.result():", future.result())

        # Update best score and parameters if the current score is better
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

# Print best parameters and score
print("Best Parameters:", best_params)
print("Best RMSE:", best_score)

# Train final model with best parameters
final_model = xgb.XGBRegressor(
    **best_params
)
final_model.fit(X_train_processed, y_train_processed)


model = final_model

# Save the model
joblib.dump(model, 'xgboost-model.pkl')

# Predict on X_test.csv, the format should be Id, 單價元平方公尺, and the number should be a float with one decimal place
X_test = pd.read_csv('data/X_test.csv')
X_test_processed = pipeline.transform(X_test)
y_test = model.predict(X_test_processed)
y_test_df = pd.DataFrame(y_test, columns=['單價元平方公尺'])
y_test_df['Id'] = X_test['Id']
y_test_df = y_test_df[['Id', '單價元平方公尺']]
# print to float with one decimal place
y_test_df['單價元平方公尺'] = y_test_df['單價元平方公尺'].round(1)
y_test_df.to_csv('data/y_test.csv', index=False)
