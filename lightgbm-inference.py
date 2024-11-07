import pandas as pd
import joblib

# read pipeline from xgboost-pipeline.pkl
pipeline = joblib.load('lightgbm-pipeline.pkl')

# read xgboost model
model = joblib.load('lgb-model.pkl')

X_test = pd.read_csv('data/X_test.csv')
X_test_processed = pipeline.transform(X_test)
y_test = model.predict(X_test_processed)
y_test_df = pd.DataFrame(y_test, columns=['單價元平方公尺'])
y_test_df['Id'] = X_test['Id']
y_test_df = y_test_df[['Id', '單價元平方公尺']]
# print to float with one decimal place
y_test_df['單價元平方公尺'] = y_test_df['單價元平方公尺'].round(1)
y_test_df.to_csv('data/y_test.csv', index=False)
