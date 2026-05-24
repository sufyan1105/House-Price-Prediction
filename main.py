from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# Load the dataset
housing = pd.read_csv("housing.csv")

# Split the dataset into training and testing sets
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):     
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)     
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()

# Separate features and target variable
houing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1) 

# seperate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# Create pipelines for numerical and categorical attributes

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Construct a full pipeline 
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Transform the data using the full pipeline
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# Train a Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(housing_prepared, houing_labels)
lin_preds = linear_regressor.predict(housing_prepared)
# lin_mse = root_mean_squared_error(houing_labels, lin_preds)
lin_mse = -cross_val_score(linear_regressor, housing_prepared, houing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print("Linear Regression RMSE:", lin_mse)
print(pd.Series(lin_mse).describe())


# Train a Decision Tree Regressor
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(housing_prepared, houing_labels)
tree_preds = tree_regressor.predict(housing_prepared)
# tree_mse = root_mean_squared_error(houing_labels, tree_preds)
tree_rmse = -cross_val_score(tree_regressor, housing_prepared, houing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print("Decision Tree RMSE:", tree_rmses)
print(pd.Series(tree_rmse).describe())

# Train a Random Forest Regressor
forest_regressor = RandomForestRegressor(random_state=42)
forest_regressor.fit(housing_prepared, houing_labels)
forest_preds = forest_regressor.predict(housing_prepared)
# forest_mse = root_mean_squared_error(houing_labels, forest_preds)
# print("Random Forest RMSE:", forest_mse)
forest_rmse = -cross_val_score(forest_regressor, housing_prepared, houing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(forest_rmse).describe())