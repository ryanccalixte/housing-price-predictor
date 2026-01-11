# libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#data (make path relative to this script)
housing_file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
housing_data = pd.read_csv(housing_file_path)

# preparing model values
y = housing_data.Price
features = ['SqFt', 'Bedrooms','Bathrooms','Brick','Neighborhood']
x = housing_data[features]
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

# pipeline preprocessing for categorical and numerical data
categorical_cols = [cname for cname in train_x.columns if train_x[cname].dtype == "object"]
numerical_cols = [cname for cname in train_x.columns if train_x[cname].dtype in ['int64', 'float64']]
numerical_transformer = SimpleImputer(strategy='constant')


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# model
forest_model = RandomForestRegressor(random_state=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', forest_model)
                             ])

pipeline.fit(train_x, train_y)

# predictions
predictions = pipeline.predict(val_x)

# create a tidy results table (reset indexes to align rows)
val_x_reset = val_x.reset_index(drop=True)
val_y_reset = val_y.reset_index(drop=True)
results = val_x_reset.copy()
results['Actual Price'] = val_y_reset
results['Predicted Price'] = predictions
results['Abs Error'] = (results['Actual Price'] - results['Predicted Price']).abs()
results['Pct Error'] = (results['Abs Error'] / results['Actual Price']).replace([float('inf'), float('nan')], 0.0)

# prepare a display-friendly copy with formatted currency
display_df = results.copy()
for col in ['Actual Price', 'Predicted Price', 'Abs Error']:
    display_df[col] = display_df[col].apply(lambda v: f"${v:,.0f}")
display_df['Pct Error'] = display_df['Pct Error'].apply(lambda v: f"{v*100:,.1f}%")

# show the first 10 predictions in a nice table
print("\nPrediction results (first 10 rows):")
print(display_df.head(10).to_string(index=False))

# summary metrics
mae = mean_absolute_error(val_y, predictions)
mean_pct_error = (results['Pct Error']).mean()
print(f"\nSummary: Mean Absolute Error = ${mae:,.0f} | Mean Percentage Error = {mean_pct_error*100:,.2f}%")