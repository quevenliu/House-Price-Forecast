# House Prediction Project

Please use virtual environment to run this project in order to reproduce the results.

## Development Environment

Windows 11 with support for PowerShell. (The virtual environment setup in this project have to be run based on PowerShell scripts.)

## How to enter virtual environment

The project is using venv for virtual environment. To enter the virtual environment, please run the following command in the project root directory.

```bash
./.venv/Scripts/Activate.ps1
```

## How to install dependencies

Once enter the virtual environment, please run the following command to install all the dependencies.

```bash
pip install -r requirements.txt
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
