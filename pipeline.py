import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


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
        # ('gbf', GBFGenerator(K=5), ['橫坐標', '縱坐標']), # Uncomment this for experiment on GBF
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
