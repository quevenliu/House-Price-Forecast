# House Prediction Project

Please use virtual environment to run this project in order to reproduce the results.

## Development Environment
Python 3.11.4, please download it [here](https://www.python.org/downloads/release/python-3114/).

Update your pip to the newest version by
```bash
py -3.11 -m pip install --upgrade pip --user
```

Please also install pipenv first by
```bash
pip install pipenv
```

## How to install dependencies

Please run the following command to install all the dependencies.

```bash
pipenv install
```

## Enter into virtual environment

Please run the following command to enter into virtual environment.
```bash
pipenv shell
```

## How to run the project

First, you should put the data files in `./data`. With name being, `X_train.csv`, `X_test.csv`, `y_train.csv`.

To run the full training and grid searching process, please run the following command for XGBoost.

```bash
python xgboost-train.py
```

And the following for LightGBM.

```bash
python lightgbm-train.py
```

To only run inference based on the pickle file provided in the repository, please run the following command for XGBoost.

```bash
python xgboost-inference.py
```

And the following for LightGBM.

```bash
python lightgbm-inference.py
```
