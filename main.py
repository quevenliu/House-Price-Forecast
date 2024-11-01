import joblib
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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
columns_to_normalize = ['土地移轉總面積平方公尺', '土地數', '建物數', '車位數', '總樓層數', '建物移轉總面積平方公尺',
                        '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛', '地鐵站', '超商', '公園',
                        '托兒所', '國小', '國中', '高中職', '大學', '金融機構', '醫院', '大賣場', '超市',
                        '百貨公司', '警察局', '消防局', '縱坐標', '橫坐標', '屋齡']

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

# Fit and transform the training data
X_train_processed = pipeline.fit_transform(X_train)

# y_train is a DataFrame, consists of Id and 單價元平方公尺, we need to preprocess it to a numpy array
y_train_processed = y_train['單價元平方公尺'].values

# Train the model
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train_processed, y_train_processed)

# Save the model
joblib.dump(model, 'model.pkl')

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
