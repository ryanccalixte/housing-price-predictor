# Basic Decision Tree Project

A small example project that trains a Decision Tree regressor on housing data and reports prediction results and the model's mean absolute error.

## Contents
- `basiclearningdata.py` — trains the model, evaluates different tree sizes, and prints predictions.
- `Housing.csv` — expected dataset (place in the same folder or update the path in the script).

## Requirements
- Python 3.8+ (tested)
- pandas
- scikit-learn

Install dependencies with:

```powershell
pip install pandas scikit-learn
```

# Advanced Housing Price Project

This project trains and evaluates a Random Forest regression model to predict house prices from a small tabular dataset. It demonstrates a preprocessing pipeline (numeric imputation + one-hot encoding), model training, and a user-friendly printed output of predictions and summary metrics.

## Contents
- `advancedhousingprice.py` — main script that prepares the data, trains a pipeline (preprocessor + `RandomForestRegressor`), prints a formatted prediction table and summary metrics.
- `HousingData.csv` — dataset (expected in the same folder as the script).

## Features used
- `SqFt`, `Bedrooms`, `Bathrooms`, `Brick`, `Neighborhood`

## What the script does
- Loads `HousingData.csv`
- Splits data into train and validation sets
- Builds a preprocessing pipeline: numeric imputation and one-hot encoding for categorical columns
- Trains a `RandomForestRegressor` inside a `Pipeline`
- Outputs a human-friendly table showing actual vs predicted prices, absolute and percentage errors, and prints Mean Absolute Error and mean percentage error


## Usage
1. Place `HousingData.csv` in the `advancedhousingpriceproject` folder (or update the `housing_file_path` in `advancedhousingprice.py`).
2. Run the project using a python powershell command or through a source code editor

The script prints the first 10 rows of prediction results in a formatted table and a short summary with Mean Absolute Error.

## Output example
- A table with columns: input features, `Actual Price`, `Predicted Price`, `Abs Error`, `Pct Error` (formatted as currency and percent)
- Summary line: `Mean Absolute Error = $... | Mean Percentage Error = ...%`

## Tips
- If you want to reproduce results exactly, set a fixed random seed where needed (the script uses `random_state=0`).
- To capture your environment, run `pip freeze > requirements.txt` and add it to the repo.
