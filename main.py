from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
import pandas as pd

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
print(housing_prepared)


