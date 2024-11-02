import joblib
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tqdm import tqdm

# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Custom Transformer for Date Columns and Calculations


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['建築完成年月'] = pd.to_datetime(X['建築完成年月'])
        X['交易日期'] = pd.to_datetime(X['交易年'].astype(
            str) + '-' + X['交易月'].astype(str) + '-' + X['交易日'].astype(str))
        X['屋齡'] = (X['交易日期'] - X['建築完成年月']).dt.days
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
